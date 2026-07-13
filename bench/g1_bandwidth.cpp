/* G1 — the gate for the zero-copy expert cache (ARCHITECTURE.md).
 *
 * Measures the 8060S read bandwidth from HOST-registered (GTT, unified) memory
 * vs from VRAM (device-local). The whole "no duplicate copies" thesis rests on
 * host-GTT reads being close to VRAM. If they are, experts stay zero-copy in the
 * shared pool; if not, hot experts keep a VRAM mirror.
 *
 * Build on the target (ROCm):  hipcc -O3 bench/g1_bandwidth.cpp -o bench/g1
 * Run:                         ./bench/g1 [buffer_MB]     (default 2048)
 * Not compiled in the dev sandbox (no ROCm/GPU there).
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  std::fprintf(stderr,"%s:%d %s -> %s\n",__FILE__,__LINE__,#x,hipGetErrorString(e)); std::exit(1);} }while(0)

__global__ void reduce_read(const float* __restrict__ p, size_t n, double* out){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x*blockDim.x;
    float s=0.f;
    for(; i<n; i+=stride) s += p[i];
    __shared__ float sh[256];
    sh[threadIdx.x]=s; __syncthreads();
    for(int k=blockDim.x>>1;k;k>>=1){ if(threadIdx.x<k) sh[threadIdx.x]+=sh[threadIdx.x+k]; __syncthreads(); }
    if(!threadIdx.x) atomicAdd(out,(double)sh[0]);
}

static double bw_gbps(const float* dptr, size_t n, int iters){
    double* d_out; CK(hipMalloc(&d_out,sizeof(double)));
    CK(hipMemset(d_out,0,sizeof(double)));
    dim3 grid(1024), block(256);
    hipLaunchKernelGGL(reduce_read,grid,block,0,0,dptr,n,d_out); CK(hipDeviceSynchronize()); /* warmup */
    hipEvent_t a,b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    CK(hipEventRecord(a));
    for(int it=0;it<iters;it++) hipLaunchKernelGGL(reduce_read,grid,block,0,0,dptr,n,d_out);
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms=0; CK(hipEventElapsedTime(&ms,a,b)); CK(hipFree(d_out));
    return (double)n*sizeof(float)*iters / (ms/1e3) / 1e9;
}

int main(int argc,char**argv){
    size_t MB = argc>1 ? strtoull(argv[1],0,10) : 2048;
    size_t n  = MB*1024ull*1024ull/sizeof(float);
    int iters = 20;
    hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr,0));
    std::printf("device: %s | buffer %zu MB | %d iters\n", pr.name, MB, iters);

    float* dv; CK(hipMalloc(&dv,n*sizeof(float))); CK(hipMemset(dv,0,n*sizeof(float)));
    double vram=bw_gbps(dv,n,iters); CK(hipFree(dv));

    float* hp=(float*)std::malloc(n*sizeof(float));
    for(size_t i=0;i<n;i++) hp[i]=1.0f;
    CK(hipHostRegister(hp,n*sizeof(float),hipHostRegisterMapped));
    float* dp; CK(hipHostGetDevicePointer((void**)&dp,hp,0));
    double gtt=bw_gbps(dp,n,iters);
    CK(hipHostUnregister(hp)); std::free(hp);

    std::printf("\n  VRAM   read: %6.1f GB/s\n", vram);
    std::printf("  host-GTT read: %6.1f GB/s  (%.0f%% of VRAM)\n", gtt, 100.0*gtt/vram);
    std::printf("\nG1 verdict: %s\n",
        gtt>=0.70*vram ? "ZERO-COPY viable -> map experts host-resident, no VRAM mirror."
                       : "KEEP a VRAM MIRROR for hot experts -> host-GTT read too slow for pure zero-copy.");
    return 0;
}
