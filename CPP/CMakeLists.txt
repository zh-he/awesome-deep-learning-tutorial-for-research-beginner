cmake_minimum_required(VERSION 3.10)
project(CPP)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 CUDA
find_package(CUDAToolkit)

# 设置 LibTorch 的路径
set(Torch_DIR "E:/Libtorch/libtorch/share/cmake/Torch")

# 找到 LibTorch
find_package(Torch REQUIRED)

# 添加可执行目标
add_executable(lenet_mnist 04_advanced_dl.cpp)

# 链接 LibTorch 库
target_link_libraries(lenet_mnist "${TORCH_LIBRARIES}")

# 如果找到了 CUDA，我们将 CUDA 库也链接到目标上
if (CUDAToolkit_FOUND)
  target_link_libraries(lenet_mnist CUDA::cudart CUDA::cuda_driver)
endif()

# 设置目标的 C++ 标准
set_property(TARGET lenet_mnist PROPERTY CXX_STANDARD 17)
