/* Device Backend seam — ARCHITECTURE.md subsystem 4.
 *
 * One name for the GPU control plane regardless of vendor, so glm.c is written
 * once: CUDA (discrete, copy-based) or HIP/ROCm (Strix Halo iGPU). G1 on the
 * 8060S measured host-GTT at 55% of VRAM, so the engine uses the mirror/upload
 * path (coli_gpu_matmul) for hot resident tensors; the zero-copy map path
 * (coli_gpu_map/matmul_mapped, HIP only) is reserved for cold streamed experts.
 * Select at build time: CUDA=1 or HIP=1. CPU stays the default. */
#ifndef COLIBRI_BACKEND_H
#define COLIBRI_BACKEND_H

#if defined(COLI_HIP)
  #include "backend_hip.h"
  #define COLI_GPU 1
  #define COLI_GPU_MAX_DEVICES COLI_HIP_MAX_DEVICES
  typedef ColiHipTensor ColiGpuTensor;
  #define coli_gpu_init          coli_hip_init
  #define coli_gpu_shutdown      coli_hip_shutdown
  #define coli_gpu_device_count  coli_hip_device_count
  #define coli_gpu_device_at     coli_hip_device_at
  #define coli_gpu_mem_info      coli_hip_mem_info
  #define coli_gpu_stats         coli_hip_stats
  #define coli_gpu_tensor_upload coli_hip_tensor_upload
  #define coli_gpu_matmul        coli_hip_matmul
  #define coli_gpu_tensor_free   coli_hip_tensor_free
  #define coli_gpu_tensor_bytes  coli_hip_tensor_bytes
  #define coli_gpu_tensor_device coli_hip_tensor_device
  /* zero-copy (unified memory) — HIP only */
  #define coli_gpu_map           coli_hip_map
  #define coli_gpu_unmap         coli_hip_unmap
  #define coli_gpu_matmul_mapped coli_hip_matmul_mapped
  /* Upstream (2026-07) CUDA-only APIs the HIP backend does not implement yet:
   * inert fallbacks so shared COLI_GPU code compiles. tensor_update FAILS (0)
   * so REPIN's VRAM-refresh path is never taken on HIP; group stats read 0. */
  #include <stdint.h>
  static inline int coli_cuda_tensor_update(ColiHipTensor *t, const void *w, const float *s){
      (void)t; (void)w; (void)s; return 0;
  }
  static inline void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                           double *h2d, double *kernel, double *d2h){
      if(calls)*calls=0; if(experts)*experts=0; if(rows)*rows=0;
      if(h2d)*h2d=0; if(kernel)*kernel=0; if(d2h)*d2h=0;
  }

#elif defined(COLI_CUDA)
  #include "backend_cuda.h"
  #define COLI_GPU 1
  #define COLI_GPU_MAX_DEVICES COLI_CUDA_MAX_DEVICES
  typedef ColiCudaTensor ColiGpuTensor;
  #define coli_gpu_init          coli_cuda_init
  #define coli_gpu_shutdown      coli_cuda_shutdown
  #define coli_gpu_device_count  coli_cuda_device_count
  #define coli_gpu_device_at     coli_cuda_device_at
  #define coli_gpu_mem_info      coli_cuda_mem_info
  #define coli_gpu_stats         coli_cuda_stats
  #define coli_gpu_tensor_upload coli_cuda_tensor_upload
  #define coli_gpu_matmul        coli_cuda_matmul
  #define coli_gpu_tensor_free   coli_cuda_tensor_free
  #define coli_gpu_tensor_bytes  coli_cuda_tensor_bytes
  #define coli_gpu_tensor_device coli_cuda_tensor_device
  /* discrete GPU: no unified pool -> zero-copy maps are no-ops (glm.c uses the
   * upload path on CUDA; matmul_mapped is never called on this backend) */
  #define coli_gpu_map(p,b,d)    ((void*)0)
  #define coli_gpu_unmap(p)      ((void)0)
#endif

#endif
