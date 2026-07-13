/* HIP/ROCm backend for the Radeon 8060S (gfx1151) on Strix Halo.
 *
 * Mirrors backend_cuda.h (copy/upload path = the "VRAM mirror" mode) and ADDS a
 * zero-copy path for unified memory: map an already-host-resident expert slab
 * into the iGPU (hipHostRegister + device pointer) and let the kernel read it in
 * place, with no duplicate VRAM allocation. Which mode the engine uses is the G1
 * decision (bench/g1_bandwidth.cpp). Build with `make HIP=1`. */
#ifndef COLIBRI_BACKEND_HIP_H
#define COLIBRI_BACKEND_HIP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_HIP_MAX_DEVICES 16

typedef struct ColiHipTensor ColiHipTensor;

int  coli_hip_init(const int *devices, int count);
void coli_hip_shutdown(void);
int  coli_hip_device_count(void);
int  coli_hip_device_at(int index);
int  coli_hip_mem_info(int device, size_t *free_bytes, size_t *total_bytes);
void coli_hip_stats(int device, size_t *tensor_count, size_t *tensor_bytes);

/* ---- VRAM-mirror mode (identical contract to CUDA: upload once, reuse) ---- */
int coli_hip_tensor_upload(ColiHipTensor **tensor,
                           const void *weights, const float *scales,
                           int fmt, int I, int O, int device);
int coli_hip_matmul(ColiHipTensor **tensor,
                    float *y, const float *x,
                    const void *weights, const float *scales,
                    int fmt, int S, int I, int O, int device);
void   coli_hip_tensor_free(ColiHipTensor *tensor);
size_t coli_hip_tensor_bytes(const ColiHipTensor *tensor);
int    coli_hip_tensor_device(const ColiHipTensor *tensor);

/* ---- Zero-copy mode (Strix Halo unified memory) ----
 * map: pin a host slab (the ExpertCache buffer) and return its device-visible
 * pointer; call once per slab at load time. unmap at eviction. No bytes copied.
 * matmul_mapped: run the matmul reading weights straight from that mapped host
 * pointer (dev_weights/dev_scales from coli_hip_map). */
void *coli_hip_map(void *host_ptr, size_t bytes, int device);   /* -> device ptr, or NULL */
void  coli_hip_unmap(void *host_ptr);
int   coli_hip_matmul_mapped(float *y, const float *x,
                             const void *dev_weights, const float *dev_scales,
                             int fmt, int S, int I, int O, int device);

#ifdef __cplusplus
}
#endif

#endif
