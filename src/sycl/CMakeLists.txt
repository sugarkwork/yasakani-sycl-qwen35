if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "The experimental SYCL backend is configured for native Windows x64")
endif()
if(NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC" OR
   NOT CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|IntelLLVM)$")
  message(FATAL_ERROR "Use SYCL-enabled clang-cl or icx-cl from an x64 Visual Studio developer shell")
endif()

if(YSKN_Q35_KV5)
  message(FATAL_ERROR "The SYCL backend supports plain packs and BF16 KV only")
endif()
set(YSKN_Q35_HIDDEN "2560" CACHE STRING "SYCL Qwen3.5 text hidden size (2560 for 4B or 4096 for 9B)")
if(NOT YSKN_Q35_HIDDEN MATCHES "^(2560|4096)$")
  message(FATAL_ERROR "The SYCL backend supports Qwen3.5-4B (2560) and Qwen3.5-9B (4096)")
endif()

set(YSKN_SYCL_CUDA_ARCH "sm_120" CACHE STRING "NVIDIA architecture for the SYCL build")
find_package(CUDAToolkit REQUIRED)
set(sycl_flags
  -fsycl -fsycl-targets=nvptx64-nvidia-cuda
  "SHELL:-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${YSKN_SYCL_CUDA_ARCH}"
  "--cuda-path=${CUDAToolkit_LIBRARY_ROOT}")
add_library(yskn_sycl SHARED qwen35.cpp)
target_compile_features(yskn_sycl PRIVATE cxx_std_17)
target_include_directories(yskn_sycl PRIVATE .. ../../include ../../third_party)
target_compile_definitions(yskn_sycl PRIVATE YSKN_BUILD YSKN_Q35_HIDDEN=${YSKN_Q35_HIDDEN}
  NOMINMAX _CRT_SECURE_NO_WARNINGS)
target_compile_options(yskn_sycl PRIVATE ${sycl_flags} /utf-8 /EHsc /fp:precise)
# Use the SYCL driver for the final device link. A direct MSVC linker call
# would skip bundling device images and the DPC++ runtime.
string(JOIN " " sycl_link_flags
  -fsycl -fsycl-targets=nvptx64-nvidia-cuda
  "-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${YSKN_SYCL_CUDA_ARCH}"
  "\"--cuda-path=${CUDAToolkit_LIBRARY_ROOT}\"")
get_filename_component(sycl_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
target_link_directories(yskn_sycl PRIVATE "${sycl_bin}/../lib")
set(CMAKE_CXX_CREATE_SHARED_LIBRARY
    "<CMAKE_CXX_COMPILER> /LD ${sycl_link_flags} <OBJECTS> /Fe:<TARGET> /link /IMPLIB:<TARGET_IMPLIB> <LINK_FLAGS> <LINK_LIBRARIES>")
