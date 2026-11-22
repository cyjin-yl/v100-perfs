#!/usr/bin/env bash
ARCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_INSTALL_PATH="$ARCH_ROOT/cuda"
export CUDAToolkit_ROOT="$ARCH_ROOT/cuda"
export GPGPUSIM_ROOT="$ARCH_ROOT/accel-sim-framework/gpu-simulator/gpgpu-sim"
export ACCELSIM_ROOT="$ARCH_ROOT/accel-sim-framework/gpu-simulator"
export ACCELSIM_CONFIG=release
export GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN=1
export ACCELSIM_SETUP_ENVIRONMENT_WAS_RUN=1
export PATH="$ARCH_ROOT/build-deps/debian/usr/bin:$ARCH_ROOT/cuda/bin:$PATH"
export BISON_PKGDATADIR="$ARCH_ROOT/build-deps/debian/usr/share/bison"
export M4="$ARCH_ROOT/build-deps/debian/usr/bin/m4"
export LIBRARY_PATH="$ARCH_ROOT/build-deps/debian/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="$ARCH_ROOT/build-deps/debian/usr/include:${CPLUS_INCLUDE_PATH:-}"
SIM_CMAKE="$ARCH_ROOT/build-deps/python/cmake/data/bin/cmake"
