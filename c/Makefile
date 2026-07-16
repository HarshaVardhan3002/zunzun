UNAME_S := $(shell uname -s)
MINGW    := $(findstring MINGW,$(UNAME_S))
MSYS     := $(findstring MSYS,$(UNAME_S))
IS_WIN   := $(MINGW)$(MSYS)

ifeq ($(UNAME_S),Darwin)
# --- macOS / Apple Silicon ---
# Apple clang non include il runtime OpenMP: se c'e' libomp di Homebrew lo usa
# (brew install libomp), altrimenti compila single-thread (i pragma omp sono ignorati).
# Niente -march: su arm64 NEON e' baseline (i kernel __ARM_NEON si attivano da soli).
CC      = clang
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(OMPDIR),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
else
$(warning libomp not found: building single-threaded. For multithreading: brew install libomp)
OMPC    =
OMPL    =
endif
CFLAGS  = -O3 $(OMPC) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm $(OMPL)
EXE     =
else ifneq ($(IS_WIN),)
# --- Windows 11 x86-64 (MinGW-w64 / MSYS2) ---
# GCC + libgomp + winpthreads: pthread, OpenMP, clock_gettime, opendir/readdir,
# AVX2 intrinsics — tutto gratis, nessun porting.
# ARCH default = native (come Linux: attiva AVX-512 VNNI dove c'e', es. Zen 5).
# Per un binario PORTABILE da distribuire: make portable (= ARCH=x86-64-v3, solo AVX2).
CC      = gcc
ARCH   ?= native
CFLAGS  = -D_FILE_OFFSET_BITS=64 -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -static
EXE     = .exe
else
# --- Linux x86-64 (percorso originale, invariato) ---
CC      = gcc
# ARCH=native -> ottimizzato per QUESTA macchina (default, piu' veloce).
# ARCH=x86-64-v3 -> binario PORTABILE su qualsiasi x86-64 moderno con AVX2 (per distribuire).
# ARCH=x86-64 -> massima compatibilita' (niente AVX2: usa il path scalare di fallback).
ARCH   ?= native
CFLAGS  = -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp
EXE     =
endif

# CUDA=1 adds an opt-in backend for resident tensors. The default build remains
# pure C and keeps the original zero-dependency runtime.
CUDA      ?= 0
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler=-Wall,-Wextra
PYTHON    ?= python3
CUDA_OBJ  =
TEST_BINS = tests/test_json$(EXE) tests/test_st$(EXE) tests/test_tier$(EXE) tests/test_grammar$(EXE) tests/test_cache$(EXE) tests/test_sched$(EXE) tests/test_profile$(EXE) tests/test_sample$(EXE)
ifeq ($(CUDA),1)
ifeq ($(UNAME_S),Darwin)
$(error CUDA=1 is supported only on Linux)
endif
ifneq ($(IS_WIN),)
# GPU: stub only in Phase 1 (G0). G1 builds coli_cuda.dll with MSVC+nvcc.
$(error CUDA=1 on Windows requires G1: build coli_cuda.dll with MSVC+nvcc (see PORT_WINDOWS_PLAN.md §8))
endif
CFLAGS   += -DCOLI_CUDA
LDFLAGS  += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lstdc++
CUDA_OBJ  = backend_cuda.o
endif

# HIP=1 adds the ROCm/HIP backend for the Strix Halo iGPU (Radeon 8060S, gfx1151),
# with the zero-copy expert path. Opt-in; the default build stays pure C.
HIP        ?= 0
HIPCC      ?= hipcc
HIP_ARCH   ?= gfx1151
# ROCm install root (has lib/amdhip64.lib, bin/amdhip64.dll). Override: make HIP=1 ROCM_PATH=...
ROCM_PATH  ?= C:/Program Files/AMD/ROCm/7.1
HIPCCFLAGS ?= -O3 -std=c++17 --offload-arch=$(HIP_ARCH) -Wno-unused-result
HIP_OBJ    =
ifeq ($(HIP),1)
CFLAGS   += -DCOLI_HIP
LDFLAGS  += -L"$(ROCM_PATH)/lib" -l:amdhip64.lib -lstdc++   # -l: = exact filename (MinGW ld needs it for ROCm .lib)
HIP_OBJ   = backend_hip.o
endif

all: glm$(EXE)

# phony 'glm' → 'glm.exe' on Windows (so 'make glm' and 'zun build' work on every platform)
glm: glm$(EXE)

glm$(EXE): glm.c st.h json.h tok.h tok_unicode.h compat.h grammar.h model.h cache.h sched.h backend.h profile.h $(CUDA_OBJ) $(HIP_OBJ)
	$(CC) $(CFLAGS) glm.c $(CUDA_OBJ) $(HIP_OBJ) -o glm$(EXE) $(LDFLAGS)

backend_cuda.o: backend_cuda.cu backend_cuda.h
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) -c backend_cuda.cu -o $@

backend_hip.o: backend_hip.cpp backend_hip.h
	@command -v $(HIPCC) >/dev/null 2>&1 || { echo "hipcc not found: install ROCm or set HIPCC" >&2; exit 1; }
	$(HIPCC) $(HIPCCFLAGS) -c backend_hip.cpp -o $@

cuda-test: backend_cuda.cu backend_cuda.h tests/test_backend_cuda.cu
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) backend_cuda.cu tests/test_backend_cuda.cu -o backend_cuda_test$(EXE)
	./backend_cuda_test$(EXE)

olmoe$(EXE): olmoe.c st.h json.h compat.h
	$(CC) $(CFLAGS) olmoe.c -o olmoe$(EXE) $(LDFLAGS)

# binario portabile da distribuire su altre macchine x86-64
portable:
	$(MAKE) glm$(EXE) ARCH=x86-64-v3

iobench$(EXE): iobench.c compat.h
	$(CC) $(CFLAGS) iobench.c -o iobench$(EXE) $(LDFLAGS)

tests/test_json$(EXE): tests/test_json.c json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st$(EXE): tests/test_st.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tier$(EXE): tests/test_tier.c tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_grammar$(EXE): tests/test_grammar.c grammar.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_cache$(EXE): tests/test_cache.c cache.h
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)

tests/test_sched$(EXE): tests/test_sched.c scheduler.h
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)

tests/test_profile$(EXE): tests/test_profile.c profile.h
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)

tests/test_sample$(EXE): tests/test_sample.c sample.h
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)

test-c: $(TEST_BINS)
	@for test in $(TEST_BINS); do ./$$test || exit 1; done

test-python:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

test: test-c test-python

# integrity: catch mount/editor desync corruption (NUL bytes, truncation) + compile C
integrity:
	$(PYTHON) ../scripts/check_integrity.py

# Local validation: one portable CPU build and dependency-free tests.
check:
	$(MAKE) clean
	$(MAKE) portable
	$(MAKE) test
	$(MAKE) integrity

clean:
	rm -f olmoe$(EXE) glm$(EXE) iobench$(EXE) backend_cuda.o backend_hip.o backend_cuda_test$(EXE) $(TEST_BINS)
	rm -rf tests/__pycache__

.PHONY: all glm cuda-test portable test-c test-python test check clean integrity
