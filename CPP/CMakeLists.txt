cmake_minimum_required(VERSION 3.10)
project(dl LANGUAGES CXX CUDA)

# Set C++ standard globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MSVC specific settings
if(MSVC)
    # Force MSVC to accept the /std:c++17 flag
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
    add_compile_options(
        /utf-8      # Add UTF-8 support
        /MP         # Enable parallel compilation
        /W4         # High warning level
    )
endif()

# Set PyTorch path
set(Torch_DIR "your_torch_path")

# Find PyTorch
find_package(Torch REQUIRED)

# Add executable
add_executable(lenet_mnist "04_advanced.cpp")

# Link libraries
target_link_libraries(lenet_mnist PRIVATE ${TORCH_LIBRARIES})

# Set target properties
set_target_properties(lenet_mnist PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Copy PyTorch DLLs to output directory (Windows only)
if(MSVC)
    add_custom_command(TARGET lenet_mnist POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE_DIR:torch_cpu>
        $<TARGET_FILE_DIR:lenet_mnist>
    )
endif()