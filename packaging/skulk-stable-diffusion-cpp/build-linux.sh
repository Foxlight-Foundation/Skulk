#!/usr/bin/env bash
# Run only in a pinned Ubuntu 22.04 developer container, never on model use.
set -euo pipefail

native_variant=$1
native_platform_tag=$2
native_gpu_targets=${3:-}
native_workspace=/work
cd "$native_workspace"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq -o APT::Update::Error-Mode=any -o Acquire::Retries=3
apt-get install -y -qq -o Acquire::Retries=3 build-essential git curl ca-certificates python3 python3-pip pkg-config patchelf xz-utils binutils > native-dependencies.log 2>&1
python3 -m pip install --disable-pip-version-check cmake
git config --global --add safe.directory /work/native-source
git config --global --add safe.directory /work/native-source/ggml

native_flags=(-DCMAKE_BUILD_TYPE=Release -DSD_SERVER_BUILD_FRONTEND=OFF -DSD_RPC=OFF -DSD_WEBM=OFF -DSD_WEBP=ON -DGGML_NATIVE=OFF)
case "$native_variant" in
  cuda)
    export PATH="/usr/local/cuda/bin:$PATH"
    native_flags+=(-DSD_CUDA=ON -DGGML_STATIC=ON -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_NO_VMM=ON "-DCMAKE_CUDA_ARCHITECTURES=$native_gpu_targets")
    if [[ "$native_platform_tag" == *_aarch64 ]]; then
      native_flags+=(-DGGML_CPU_ARM_ARCH=armv8.2-a+fp16)
    fi
    ;;
  vulkan)
    apt-get install -y -qq libvulkan-dev
    curl --fail --location --retry 3 https://sdk.lunarg.com/sdk/download/1.4.350.0/linux/vulkansdk-linux-x86_64-1.4.350.0.tar.xz -o vulkan-sdk.tar.xz
    echo 'b65f068ab36263559da49d7cacd7e7b9df23824ca8b68ccc522a2b06f5725df2  vulkan-sdk.tar.xz' | sha256sum --check --strict
    tar -xf vulkan-sdk.tar.xz
    export VULKAN_SDK=/work/1.4.350.0/x86_64
    export PATH="$VULKAN_SDK/bin:$PATH"
    export LD_LIBRARY_PATH="$VULKAN_SDK/lib:${LD_LIBRARY_PATH:-}"
    native_flags+=(-DSD_VULKAN=ON -DGGML_STATIC=ON)
    ;;
  rocm)
    native_flags+=(-DSD_HIPBLAS=ON -DGGML_STATIC=OFF -DGGML_HIP_RCCL=OFF -DGGML_HIP_NO_VMM=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ "-DCMAKE_HIP_ARCHITECTURES=$native_gpu_targets" "-DGPU_TARGETS=$native_gpu_targets")
    ;;
  *) echo 'Unsupported native image build variant' >&2; exit 1 ;;
esac
cmake -S native-source -B native-build "${native_flags[@]}" > native-configure.log 2>&1
cmake --build native-build --target sd-server --parallel 3 > native-build.log 2>&1
ldd native-build/bin/sd-server > native-loader.txt
readelf --version-info native-build/bin/sd-server > native-symbols.txt
mkdir native-runtime
case "$native_variant" in
  cuda)
    # The static CMake contract embeds cudart and cuBLAS; only the driver is external.
    if grep -E 'lib(cudart|cublas|cublasLt|nccl)\.so' native-loader.txt; then exit 1; fi
    for native_license in /usr/share/doc/libcublas-*/copyright /usr/share/doc/cuda-cudart-*/copyright; do
      native_component=$(basename "$(dirname "$native_license")")
      cp "$native_license" "native-runtime/$native_component-COPYRIGHT.txt"
    done
    ;;
  vulkan)
    cp -L /usr/lib/x86_64-linux-gnu/libvulkan.so.1 native-runtime/libvulkan.so.1
    cp /usr/share/doc/libvulkan1/copyright native-runtime/VULKAN_LOADER_COPYRIGHT.txt
    ;;
  rocm)
    # Runtime bundling is separate from the native source pin and preserves the
    # exact compiler-image libraries and GPU code objects in payload provenance.
    python3 scripts/bundle_stable_diffusion_cpp_rocm.py --binary native-build/bin/sd-server --output native-runtime --gpu-target "$native_gpu_targets"
    ;;
esac
patchelf --set-rpath '$ORIGIN/../lib' native-build/bin/sd-server
python3 scripts/build_stable_diffusion_cpp_wheel.py --source native-source --build native-build --output candidate-wheels --variant "$native_variant" --platform-tag "$native_platform_tag" --runtime native-runtime
