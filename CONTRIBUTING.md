# Contributing

Contributions are actively wanted — this project is one person plus whoever shows up.
The areas where help moves the needle most:

- **Benchmarks** from AMD / unified-memory boxes (Strix Halo, Ryzen AI, Apple Silicon)
  — real datapoints drive every optimization decision here;
- **Backend ports** — the GPU seam (`c/backend.h`) is deliberately small; HIP exists,
  Vulkan or Metal would widen the non-CUDA story;
- **MoE models beyond GLM-5.2** — the runtime is meant to be model-agnostic
  (see `ROADMAP.md`);
- **I/O-path experiments** — direct reads, async pipelines, prefetch predictors.

Ground rules: keep changes focused and preserve Zunzun's dependency-free default CPU
path. Zunzun builds on [colibrì](https://github.com/JustVugg/colibri) (Apache 2.0);
changes that make sense upstream are worth offering there too.

## Local checks

Run the lightweight checks locally:

```sh
make check
```

`make -C c check` remains available for scripts that already run from the
engine directory.

This performs one portable CPU build, C unit tests, and Python standard-library
tests. It does not download a model or require CUDA.

CUDA changes should additionally be checked on a CUDA-capable Linux host:

```sh
make -C c cuda-test CUDA_ARCH=native
```

Benchmark reports should include the commit, exact commands, hardware and
storage details, warm-up policy, run count, and median throughput.
