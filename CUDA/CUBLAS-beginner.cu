#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

// 错误检查宏
#define CHECK_CUDA(func) { cudaAssert((func), __FILE__, __LINE__); }
#define CHECK_CUBLAS(func) { cublasAssert((func), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void cublasAssert(cublasStatus_t code, const char *file, int line) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS Error: Code " << code
                  << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    /************************************
     * 第一部分：向量基本操作
     * 目标：掌握向量加法（AXPY）和点积（DOT）
     ************************************/
    const int VEC_SIZE = 4;
    cublasHandle_t handle;
    
    // 步骤1：创建CUBLAS句柄
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // 步骤2：在主机端初始化数据
    std::vector<float> h_x(VEC_SIZE);  // 输入向量x
    std::vector<float> h_y(VEC_SIZE);  // 输入向量y
    std::vector<float> h_result(VEC_SIZE); // 存储结果
    
    // 使用iota填充向量：x = [0,1,2,3], y = [4,5,6,7]
    std::iota(h_x.begin(), h_x.end(), 0.0f);
    std::iota(h_y.begin(), h_y.end(), 4.0f);
    
    // 步骤3：在设备端分配内存
    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, VEC_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, VEC_SIZE * sizeof(float)));
    
    // 步骤4：数据拷贝到设备
    CHECK_CUBLAS(cublasSetVector(VEC_SIZE, sizeof(float), h_x.data(), 1, d_x, 1));
    CHECK_CUBLAS(cublasSetVector(VEC_SIZE, sizeof(float), h_y.data(), 1, d_y, 1));
    
    // 操作1：向量加法 y = α*x + y
    // 数学公式：y ← αx + y
    // 对应BLAS级别：1（向量-向量操作）
    const float alpha = 2.0f;
    CHECK_CUBLAS(cublasSaxpy(handle, VEC_SIZE, &alpha, d_x, 1, d_y, 1));
    
    // 步骤5：取回结果
    CHECK_CUBLAS(cublasGetVector(VEC_SIZE, sizeof(float), d_y, 1, h_result.data(), 1));
    std::cout << "向量加法结果 y = 2*x + y:\n";
    for (float val : h_result) std::cout << val << " ";
    std::cout << "\n\n";
    
    // 操作2：向量点积
    // 数学公式：result = x·y
    // 对应BLAS级别：1
    float dot_result;
    CHECK_CUBLAS(cublasSdot(handle, VEC_SIZE, d_x, 1, d_y, 1, &dot_result));
    std::cout << "向量点积结果 x·y = " << dot_result << "\n\n";
    
    /************************************
     * 第二部分：矩阵基本操作
     * 目标：掌握矩阵乘法（GEMM）
     ************************************/
    const int M = 2;  // A的行数
    const int N = 3;  // B的列数
    const int K = 2;  // A的列数/B的行数
    
    // 步骤1：主机端矩阵初始化
    // 矩阵布局：CUBLAS默认使用列优先存储（与Fortran相同）
    std::vector<float> h_A = {1.0f, 2.0f,  // A = [1 3
                               3.0f, 4.0f,  //      2 4]
                               5.0f, 6.0f};
    
    std::vector<float> h_B = {0.5f, 1.0f,  // B = [0.5 1.5
                               1.5f, 2.0f,  //      1.0 2.0
                               2.5f, 3.0f}; //      2.5 3.0]
    
    std::vector<float> h_C(M * N, 0);  // 结果矩阵
    
    // 步骤2：设备端内存分配
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M*K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K*N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M*N * sizeof(float)));
    
    // 步骤3：数据传输到设备
    CHECK_CUBLAS(cublasSetMatrix(M, K, sizeof(float), h_A.data(), M, d_A, M));
    CHECK_CUBLAS(cublasSetMatrix(K, N, sizeof(float), h_B.data(), K, d_B, K));
    
    // 步骤4：执行矩阵乘法 C = α*A*B + β*C
    // 参数说明：
    // handle : CUBLAS句柄
    // transa, transb: 是否转置矩阵（CUBLAS_OP_N不转置，CUBLAS_OP_T转置）
    // M : 矩阵A的行数（结果矩阵C的行数）
    // N : 矩阵B的列数（结果矩阵C的列数）
    // K : 矩阵A的列数/矩阵B的行数
    // alpha : 系数α
    // A : 矩阵A设备指针，维度 lda x K （lda>=M）
    // lda : A的前导维度（行数）
    // B : 矩阵B设备指针，维度 ldb x N （ldb>=K）
    // ldb : B的前导维度（行数）
    // beta : 系数β
    // C : 结果矩阵设备指针，维度 ldc x N （ldc>=M）
    // ldc : C的前导维度（行数）
    const float gemm_alpha = 1.0f;
    const float gemm_beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置A和B
                            M, N, K,                   // 矩阵维度
                            &gemm_alpha,
                            d_A, M,                    // A的维度是MxK
                            d_B, K,                    // B的维度是KxN
                            &gemm_beta,
                            d_C, M));                  // C的维度是MxN
    
    // 步骤5：取回结果
    CHECK_CUBLAS(cublasGetMatrix(M, N, sizeof(float), d_C, M, h_C.data(), M));
    
    // 步骤6：验证结果
    std::cout << "矩阵乘法结果 C = A*B:\n";
    // 列优先存储的输出处理
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            // 列优先存储的索引计算：row + col*M
            std::cout << h_C[row + col*M] << "\t";
        }
        std::cout << "\n";
    }
    
    /************************************
     * 第三部分：高级操作演示（矩阵转置）
     ************************************/
    // 目标：计算 C = A^T * B
    std::cout << "\n矩阵转置乘法 C = A^T * B:\n";
    
    // 重新初始化C
    std::fill(h_C.begin(), h_C.end(), 0);
    CHECK_CUBLAS(cublasSetMatrix(M, N, sizeof(float), h_C.data(), M, d_C, M));
    
    // 执行转置乘法（转置A）
    CHECK_CUBLAS(cublasSgemm(handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,  // 转置A，不转置B
                            M, N, K,
                            &gemm_alpha,
                            d_A, K,                    // 转置后的A维度是KxM
                            d_B, K,
                            &gemm_beta,
                            d_C, M));
    
    // 取回并显示结果
    CHECK_CUBLAS(cublasGetMatrix(M, N, sizeof(float), d_C, M, h_C.data(), M));
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            std::cout << h_C[row + col*M] << "\t";
        }
        std::cout << "\n";
    }
    
    /************************************
     * 资源清理
     ************************************/
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(handle));
    
    return 0;
}