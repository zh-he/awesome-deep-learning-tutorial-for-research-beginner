/******************************************************************************
 * Filename : all_in_one_example.cu
 * 
 * 本示例代码演示:
 *  1. 向量加法 (vectorAdd)
 *  2. 线程索引 (blockIdx, threadIdx, blockDim)
 *  3. 矩阵乘法 (matrixMul) - 包括基础实现和使用共享内存加速的实现
 *  4. 流 (Stream) 的简单使用 - 实现并行向量加法
 *  5. CUDA 基础 API 调用 (cudaMalloc, cudaMemcpy, cudaFree, cudaDeviceSynchronize)
 *  6. 错误检查宏
 * 
 * 运行方法 (Linux) :
 *   nvcc all_in_one_example.cu -o all_in_one_example
 *   ./all_in_one_example
 *****************************************************************************/

 #include <cstdio>
 #include <cstdlib>
 #include <iostream>
 
 /**********************************************
  * 错误检查宏
  **********************************************/
 #define CHECK_CUDA(call) do {                                           \
     cudaError_t err = (call);                                           \
     if (err != cudaSuccess) {                                           \
         fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n",       \
                 cudaGetErrorString(err), (int)err, __FILE__, __LINE__);\
         exit(EXIT_FAILURE);                                             \
     }                                                                   \
 } while(0)
 
 /**********************************************
  * 1) 向量加法 Kernel
  **********************************************/
 /*
   每个线程处理单个元素:
     C[idx] = A[idx] + B[idx]
 */
 __global__ void vectorAdd(const float* A, const float* B, float* C, int n)
 {
     // 计算当前线程的全局索引
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
 
     // 边界检查
     if (idx < n) {
         C[idx] = A[idx] + B[idx];
     }
 }
 
 /**********************************************
  * 2) 矩阵乘法 Kernel (naive 实现)
  **********************************************/
 /*
   计算 C = A * B
   A维度: (M x K)
   B维度: (K x N)
   C维度: (M x N)
   每个线程负责 C 中的一个元素 (row, col),
   并通过 for 循环对 A[row,*] 与 B[*,col] 做乘加.
 */
 __global__ void matrixMulNaive(const float* A, const float* B, float* C,
                                int M, int K, int N)
 {
     // row, col 对应 C 矩阵的行列
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
 
     if(row < M && col < N) {
         float sum = 0.0f;
         for(int i=0; i<K; i++){
             sum += A[row*K + i] * B[i*N + col];
         }
         C[row*N + col] = sum;
     }
 }
 
 /**********************************************
  * 3) 矩阵乘法 Kernel (使用共享内存优化)
  **********************************************/
 /*
   思路:
    - 将A和B分块拷入共享内存, 减少反复访问全局显存
    - 每个 Block 处理 C 中的一块 (blockDim.x x blockDim.y),
      在共享内存中进行局部乘加.
 
   Tips:
    - blockDim.x == blockDim.y 假设方块,简化演示.
    - 真实应用中还要考虑边界、bank conflict 等问题.
 */
 #define BLOCK_SIZE 16
 
 __global__ void matrixMulShared(const float* A, const float* B, float* C,
                                 int M, int K, int N)
 {
     // block内的共享内存
     __shared__ float As[BLOCK_SIZE][BLOCK_SIZE]; 
     __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
 
     // 计算当前线程的全局行、列
     int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
     int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
 
     float sum = 0.0f;
 
     // 分块迭代: 每次加载 A, B 的一块到共享内存
     // 目标: 累加 A[row, i..i+BLOCK_SIZE-1] * B[i.., col]
     for(int t = 0; t < (K + BLOCK_SIZE - 1)/BLOCK_SIZE; t++){
         // 从全局内存拷贝 A的一块 -> As
         // 注: 需要检查越界 (row < M && t*BLOCK_SIZE + tx < K)
         int loadAcol = t*BLOCK_SIZE + threadIdx.x;
         if(row < M && loadAcol < K){
             As[threadIdx.y][threadIdx.x] = A[row*K + loadAcol];
         } else {
             As[threadIdx.y][threadIdx.x] = 0.0f;
         }
 
         // 从全局内存拷贝 B的一块 -> Bs
         // 注: 需要检查越界
         int loadBrow = t*BLOCK_SIZE + threadIdx.y;
         if(loadBrow < K && col < N){
             Bs[threadIdx.y][threadIdx.x] = B[loadBrow*N + col];
         } else {
             Bs[threadIdx.y][threadIdx.x] = 0.0f;
         }
 
         // 同步, 确保 As, Bs 数据已载入共享内存
         __syncthreads();
 
         // 累加乘积
         for(int i=0; i<BLOCK_SIZE; i++){
             sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
         }
 
         // block 内所有线程都要一起执行完毕后, 才能进行下一次分块拷贝
         __syncthreads();
     }
 
     // 写入结果
     if(row < M && col < N){
         C[row*N + col] = sum;
     }
 }
 
 /**********************************************
  * 4) 在主函数中演示各种 CUDA 操作
  *    - 向量加法
  *    - 矩阵乘法(naive & shared)
  *    - 多流并行示例
  **********************************************/
 
 int main()
 {
     /*****************************************************
      * Part A: 向量加法 (单流)
      *****************************************************/
     {
         std::cout << "=== Part A: 向量加法 ===\n";
 
         int n = 10;  // 向量长度(示例取小值)
         size_t size = n * sizeof(float);
 
         // Host 端分配
         float *h_A = (float*)malloc(size);
         float *h_B = (float*)malloc(size);
         float *h_C = (float*)malloc(size);
 
         // 初始化
         for(int i = 0; i < n; i++){
             h_A[i] = float(i);
             h_B[i] = float(i*2);
         }
 
         // Device 端分配
         float *d_A, *d_B, *d_C;
         CHECK_CUDA(cudaMalloc((void**)&d_A, size));
         CHECK_CUDA(cudaMalloc((void**)&d_B, size));
         CHECK_CUDA(cudaMalloc((void**)&d_C, size));
 
         // 拷贝 Host->Device
         CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
         CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
 
         // Kernel 配置
         int blockSize = 256;
         int gridSize = (n + blockSize - 1) / blockSize;
 
         // 启动 Kernel
         vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
         CHECK_CUDA(cudaGetLastError());
         CHECK_CUDA(cudaDeviceSynchronize()); // 等待 GPU 计算完成
 
         // 拷回结果
         CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
 
         // 显示
         for(int i=0; i<n; i++){
             std::cout << "C[" << i << "] = " << h_C[i] << "\n";
         }
         std::cout << std::endl;
 
         // 释放
         CHECK_CUDA(cudaFree(d_A));
         CHECK_CUDA(cudaFree(d_B));
         CHECK_CUDA(cudaFree(d_C));
         free(h_A);
         free(h_B);
         free(h_C);
     }
 
     /*****************************************************
      * Part B: 矩阵乘法 (Naive vs Shared Memory)
      *****************************************************/
     {
         std::cout << "=== Part B: 矩阵乘法 ===\n";
 
         // 定义矩阵维度
         int M = 4, K = 5, N = 3; // A(4x5), B(5x3), C(4x3)
 
         size_t sizeA = M*K*sizeof(float);
         size_t sizeB = K*N*sizeof(float);
         size_t sizeC = M*N*sizeof(float);
 
         // Host 分配
         float *h_A = (float*)malloc(sizeA);
         float *h_B = (float*)malloc(sizeB);
         float *h_C_naive = (float*)malloc(sizeC);
         float *h_C_shared= (float*)malloc(sizeC);
 
         // 初始化 A, B
         for(int i=0; i<M*K; i++){
             h_A[i] = float(i+1); // 1,2,3,...
         }
         for(int i=0; i<K*N; i++){
             h_B[i] = float((i+1)*0.5f); // 0.5,1.0,1.5,...
         }
 
         // Device 分配
         float *d_A, *d_B, *d_C_naive, *d_C_shared;
         CHECK_CUDA(cudaMalloc((void**)&d_A, sizeA));
         CHECK_CUDA(cudaMalloc((void**)&d_B, sizeB));
         CHECK_CUDA(cudaMalloc((void**)&d_C_naive, sizeC));
         CHECK_CUDA(cudaMalloc((void**)&d_C_shared, sizeC));
 
         CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
         CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
 
         // 1) Naive kernel
         {
             dim3 block(16, 16);
             dim3 grid( (N+block.x-1)/block.x, (M+block.y-1)/block.y );
 
             matrixMulNaive<<<grid, block>>>(d_A, d_B, d_C_naive, M, K, N);
             CHECK_CUDA(cudaGetLastError());
             CHECK_CUDA(cudaDeviceSynchronize());
             CHECK_CUDA(cudaMemcpy(h_C_naive, d_C_naive, sizeC, cudaMemcpyDeviceToHost));
 
             std::cout << "[Naive] C = A x B :\n";
             for(int i=0; i<M; i++){
                 for(int j=0; j<N; j++){
                     std::cout << h_C_naive[i*N + j] << " ";
                 }
                 std::cout << "\n";
             }
             std::cout << std::endl;
         }
 
         // 2) Shared Memory kernel
         {
             // 共享内存版本 blockSize = 16(见 #define)
             // grid需根据N, M计算
             dim3 block(BLOCK_SIZE, BLOCK_SIZE);
             dim3 grid( (N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE );
 
             matrixMulShared<<<grid, block>>>(d_A, d_B, d_C_shared, M, K, N);
             CHECK_CUDA(cudaGetLastError());
             CHECK_CUDA(cudaDeviceSynchronize());
             CHECK_CUDA(cudaMemcpy(h_C_shared, d_C_shared, sizeC, cudaMemcpyDeviceToHost));
 
             std::cout << "[Shared Mem] C = A x B :\n";
             for(int i=0; i<M; i++){
                 for(int j=0; j<N; j++){
                     std::cout << h_C_shared[i*N + j] << " ";
                 }
                 std::cout << "\n";
             }
             std::cout << std::endl;
         }
 
         // 释放
         CHECK_CUDA(cudaFree(d_A));
         CHECK_CUDA(cudaFree(d_B));
         CHECK_CUDA(cudaFree(d_C_naive));
         CHECK_CUDA(cudaFree(d_C_shared));
         free(h_A);
         free(h_B);
         free(h_C_naive);
         free(h_C_shared);
     }
 
     /*****************************************************
      * Part C: 利用 Stream 并行执行多个向量加法
      *         (演示异步&并行)
      *****************************************************/
     {
         std::cout << "=== Part C: 多流并行向量加法 ===\n";
         // 这里演示在两个Stream中并行处理大小相同的两段向量
 
         const int n = 1 << 20; // 大约百万级
         size_t size = n * sizeof(float);
 
         // Host 端分配
         float *h_A = (float*)malloc(size);
         float *h_B = (float*)malloc(size);
         float *h_C = (float*)malloc(size);
         for(int i=0; i<n; i++){
             h_A[i] = 1.0f; // 全1
             h_B[i] = 2.0f; // 全2
         }
 
         // Device 端分配
         float *d_A, *d_B, *d_C;
         CHECK_CUDA(cudaMalloc((void**)&d_A, size));
         CHECK_CUDA(cudaMalloc((void**)&d_B, size));
         CHECK_CUDA(cudaMalloc((void**)&d_C, size));
 
         // 分两次拷贝(演示而已), 也可以一次拷完
         CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
         CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
 
         // 创建两个流
         cudaStream_t stream1, stream2;
         CHECK_CUDA(cudaStreamCreate(&stream1));
         CHECK_CUDA(cudaStreamCreate(&stream2));
 
         // 假设我们要把 n个元素划分成前半部分、后半部分分别在两个流执行
         int halfSize = n/2;
         int blockSize = 256;
         int gridSize = (halfSize + blockSize - 1)/blockSize;
 
         // 向量加法(前半部分)交给stream1
         //   数据指针起点 d_A[0], d_B[0], d_C[0]
         vectorAdd<<<gridSize, blockSize, 0, stream1>>>(d_A, d_B, d_C, halfSize);
 
         // 向量加法(后半部分)交给stream2
         //   数据指针起点 d_A[halfSize], ...
         vectorAdd<<<gridSize, blockSize, 0, stream2>>>(d_A + halfSize,
                                                        d_B + halfSize,
                                                        d_C + halfSize,
                                                        halfSize);
 
         // 主线程可以在需要结果前再同步, 或者使用 cudaStreamSynchronize
         CHECK_CUDA(cudaStreamSynchronize(stream1));
         CHECK_CUDA(cudaStreamSynchronize(stream2));
 
         // 此时C已经计算完成
         CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
 
         // 简单验证
         // 预期结果: 全部=3.0 (1.0 + 2.0)
         for(int i=0; i<5; i++){
             std::cout << "C[" << i << "] = " << h_C[i] << "\n";
         }
         std::cout << "... \n";
         for(int i=n-5; i<n; i++){
             std::cout << "C[" << i << "] = " << h_C[i] << "\n";
         }
 
         // 清理
         CHECK_CUDA(cudaStreamDestroy(stream1));
         CHECK_CUDA(cudaStreamDestroy(stream2));
         CHECK_CUDA(cudaFree(d_A));
         CHECK_CUDA(cudaFree(d_B));
         CHECK_CUDA(cudaFree(d_C));
         free(h_A);
         free(h_B);
         free(h_C);
     }
 
     std::cout << "=== 所有示例执行完毕 ===\n";
     return 0;
 }
 