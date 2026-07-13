#include "backend_hip.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

struct ColiHipTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int tracked;
};

typedef struct {
    int device;
    float *x, *y;
    size_t x_cap, y_cap;
    size_t tensor_count, tensor_bytes;
} DeviceContext;

static DeviceContext g_ctx[COLI_HIP_MAX_DEVICES];
static int g_nctx;

static int hip_ok(hipError_t err, const char *what) {
    if (err == hipSuccess) return 1;
    std::fprintf(stderr, "[HIP] %s: %s\n", what, hipGetErrorString(err));
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

static int select_ctx(DeviceContext *ctx) {
    return ctx && hip_ok(hipSetDevice(ctx->device), "select device");
}

static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2) return (size_t)(I + 1) / 2;
    if (fmt == 3) return (size_t)(I + 3) / 4;
    return 0;
}

__device__ static float weight_at(const void *weights, int fmt, size_t row, int i) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    const uint8_t *q = base;
    if (fmt == 2) {
        uint8_t v = q[i >> 1];
        return static_cast<float>(((i & 1) ? (v >> 4) : (v & 15)) - 8);
    }
    uint8_t v = q[i >> 2];
    return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
}

__global__ static void quant_matmul(float *y, const float *x, const void *weights,
                                    const float *scales, int fmt, int S, int I, int O,
                                    size_t rb) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    for (int i = threadIdx.x; i < I; i += blockDim.x)
        sum += xs[i] * weight_at(weights, fmt, row, i);

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (!threadIdx.x)
        y[(size_t)s * O + o] = partial[0] * (fmt ? scales[o] : 1.0f);
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) hipFree(*ptr);
    *ptr = nullptr;
    *cap = 0;
    if (!hip_ok(hipMalloc(ptr, bytes), "scratch allocation")) return 0;
    *cap = bytes;
    return 1;
}

extern "C" int coli_hip_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > COLI_HIP_MAX_DEVICES) return 0;
    if (!hip_ok(hipGetDeviceCount(&available), "device discovery")) return 0;
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            std::fprintf(stderr, "[HIP] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0;
            return 0;
        }
        if (find_ctx(device)) {
            std::fprintf(stderr, "[HIP] duplicate device %d\n", device);
            g_nctx = 0;
            return 0;
        }
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        hipDeviceProp_t prop{};
        if (!hip_ok(hipGetDeviceProperties(&prop, device), "device properties")) { g_nctx = 0; return 0; }
        g_nctx++;
        std::fprintf(stderr, "[HIP] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                     device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    }
    return 1;
}

extern "C" void coli_hip_shutdown(void) {
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        if (ctx->x) hipFree(ctx->x);
        if (ctx->y) hipFree(ctx->y);
        ctx->x = ctx->y = nullptr;
        ctx->x_cap = ctx->y_cap = 0;
    }
    g_nctx = 0;
}

extern "C" int coli_hip_device_count(void) { return g_nctx; }

extern "C" int coli_hip_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_hip_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    return hip_ok(hipMemGetInfo(free_bytes, total_bytes), "memory info");
}

extern "C" void coli_hip_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

extern "C" int coli_hip_tensor_upload(ColiHipTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!tensor || !weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !scales)) return 0;
    if (*tensor) {
        ColiHipTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device;
    }
    ColiHipTensor *t = static_cast<ColiHipTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    if (!hip_ok(hipMalloc(&t->weights, t->weight_bytes), "tensor allocation") ||
        !hip_ok(hipMemcpy(t->weights, weights, t->weight_bytes, hipMemcpyHostToDevice), "tensor upload")) {
        coli_hip_tensor_free(t);
        return 0;
    }
    if (fmt) {
        if (!hip_ok(hipMalloc(&t->scales, (size_t)O * sizeof(float)), "scale allocation") ||
            !hip_ok(hipMemcpy(t->scales, scales, (size_t)O * sizeof(float), hipMemcpyHostToDevice), "scale upload")) {
            coli_hip_tensor_free(t);
            return 0;
        }
    }
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + (fmt ? (size_t)O * sizeof(float) : 0);
    *tensor = t;
    return 1;
}

extern "C" int coli_hip_matmul(ColiHipTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device) {
    if (S < 1 || !coli_hip_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0;
    ColiHipTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;
    if (!hip_ok(hipMemcpy(ctx->x, x, xb, hipMemcpyHostToDevice), "input upload")) return 0;
    dim3 grid((unsigned)O, (unsigned)S);
    quant_matmul<<<grid, 256>>>(ctx->y, ctx->x, t->weights, t->scales, fmt, S, I, O, rb);
    if (!hip_ok(hipGetLastError(), "matmul launch") ||
        !hip_ok(hipMemcpy(y, ctx->y, yb, hipMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}

extern "C" void coli_hip_tensor_free(ColiHipTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        size_t bytes = tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->weights) hipFree(tensor->weights);
    if (tensor->scales) hipFree(tensor->scales);
    std::free(tensor);
}

extern "C" size_t coli_hip_tensor_bytes(const ColiHipTensor *tensor) {
    return tensor ? tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0) : 0;
}

extern "C" int coli_hip_tensor_device(const ColiHipTensor *tensor) {
    return tensor ? tensor->device : -1;
}


/* ==== R2 ZERO-COPY ADDITIONS (Strix Halo unified memory) ==================
 * Map a host-resident expert slab into the iGPU once (hipHostRegister +
 * device pointer); the matmul kernel then reads the weights straight from that
 * host memory over Infinity Fabric, with no duplicate VRAM allocation. Only the
 * small x/y scratch lives in device memory. This is the "no duplicate copies"
 * path; whether it beats a VRAM mirror is the G1 measurement. */
struct MapEntry { void *host; void *dev; };
static MapEntry g_maps[4096];
static int g_nmap;

extern "C" void *coli_hip_map(void *host_ptr, size_t bytes, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!host_ptr || !bytes || !select_ctx(ctx)) return nullptr;
    for (int i = 0; i < g_nmap; i++) if (g_maps[i].host == host_ptr) return g_maps[i].dev;
    if (!hip_ok(hipHostRegister(host_ptr, bytes, hipHostRegisterMapped), "host register")) return nullptr;
    void *dev = nullptr;
    if (!hip_ok(hipHostGetDevicePointer(&dev, host_ptr, 0), "host map")) {
        hipHostUnregister(host_ptr);
        return nullptr;
    }
    if (g_nmap < (int)(sizeof(g_maps) / sizeof(g_maps[0]))) g_maps[g_nmap++] = { host_ptr, dev };
    return dev;
}

extern "C" void coli_hip_unmap(void *host_ptr) {
    for (int i = 0; i < g_nmap; i++) if (g_maps[i].host == host_ptr) {
        hipHostUnregister(host_ptr);
        g_maps[i] = g_maps[--g_nmap];
        return;
    }
}

extern "C" int coli_hip_matmul_mapped(float *y, const float *x,
                                      const void *dev_weights, const float *dev_scales,
                                      int fmt, int S, int I, int O, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (S < 1 || !dev_weights || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !dev_scales)) return 0;
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;
    if (!hip_ok(hipMemcpy(ctx->x, x, xb, hipMemcpyHostToDevice), "input upload")) return 0;
    dim3 grid((unsigned)O, (unsigned)S);
    quant_matmul<<<grid, 256>>>(ctx->y, ctx->x, dev_weights, dev_scales, fmt, S, I, O, rb);
    if (!hip_ok(hipGetLastError(), "matmul launch") ||
        !hip_ok(hipMemcpy(y, ctx->y, yb, hipMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}
