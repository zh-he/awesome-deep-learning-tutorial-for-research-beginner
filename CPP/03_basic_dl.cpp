/******************************************************************************
 * 04_advanced_dl.cpp
 *
 * 本文件介绍 C++ 深度学习的进阶特性： 
 * 1. 多层感知器 (MLP) 示例 (含隐藏层)
 *
 * 目标：让读者在掌握单层网络原理后，再看看如何在 C++ 中搭建多层网络，
 *       并在简化场景下学习如何利用基本的多层感知器进行分类。
 *****************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

// 第一部分：多层感知器 (MLP) 示例

// 数据生成器类
class DataGenerator {
public:
    DataGenerator(size_t n_samples = 1000) 
        : gen(std::chrono::system_clock::now().time_since_epoch().count()) {
        n_samples_ = n_samples;
    }

    // 生成二分类数据
    // 生成两个同心圆形状的数据分布
    void generate_circle_data(std::vector<std::vector<double>>& X, 
                             std::vector<std::vector<double>>& y) {
        std::normal_distribution<double> noise(0.0, 0.1);
        std::uniform_real_distribution<double> angle(0.0, 2 * M_PI);

        X.resize(n_samples_);
        y.resize(n_samples_, std::vector<double>(1));

        for (size_t i = 0; i < n_samples_; i++) {
            X[i].resize(2);
            
            // 随机决定是内圆(label=0)还是外圆(label=1)
            bool is_outer = (i%2==0);
            double r = is_outer ? 2.0 : 1.0;
            
            // 生成圆形数据
            double theta = angle(gen);
            X[i][0] = r * cos(theta) + noise(gen);  // x坐标
            X[i][1] = r * sin(theta) + noise(gen);  // y坐标
            y[i][0] = is_outer ? 1.0 : 0.0;        // 标签
        }
    }

private:
    size_t n_samples_;
    std::mt19937 gen;
};

// Sigmoid 激活函数及其导数
namespace SimpleMath
{
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }

    // Binary Cross-Entropy Loss
    double binaryCrossEntropy(const std::vector<double>& yTrue, const std::vector<double>& yPred) {
        double loss = 0.0;
        for (size_t i = 0; i < yTrue.size(); i++) {
            loss += -yTrue[i] * std::log(yPred[i] + 1e-15) - (1.0 - yTrue[i]) * std::log(1.0 - yPred[i] + 1e-15);
        }
        return loss;
    }
}

// 简易多层感知器
class MLP
{
public:
    MLP(size_t inputDim, size_t hiddenDim, size_t outputDim) {
        initLayer(weights1_, bias1_, inputDim, hiddenDim);
        initLayer(weights2_, bias2_, hiddenDim, outputDim);

        // 缓存中间结果用于反向传播
        cached_z1_.resize(hiddenDim);
        cached_a1_.resize(hiddenDim);
        cached_z2_.resize(outputDim);
        cached_a2_.resize(outputDim);
    }

    // 前向传播
    // x: [inputDim], 返回值 yPred: [outputDim]
    std::vector<double> forward(const std::vector<double>& x) {
        // 1. z1 = x * W1 + b1
        for (size_t h = 0; h < weights1_[0].size(); h++) {
            double sum = 0.0;
            for (size_t i = 0; i < x.size(); i++) {
                sum += x[i] * weights1_[i][h];
            }
            sum += bias1_[h];
            cached_z1_[h] = sum; // 用于 BP
            cached_a1_[h] = SimpleMath::sigmoid(sum);
        }

        // 2. z2 = a1 * W2 + b2
        for (size_t o = 0; o < weights2_[0].size(); o++) {
            double sum = 0.0;
            for (size_t h = 0; h < cached_a1_.size(); h++) {
                sum += cached_a1_[h] * weights2_[h][o];
            }
            sum += bias2_[o];
            cached_z2_[o] = sum;
            cached_a2_[o] = SimpleMath::sigmoid(sum);
        }

        return cached_a2_;
    }

    // 反向传播 (Binary Cross-Entropy Loss)
    // yTrue, yPred 均为 [outputDim]
    void backward(const std::vector<double>& x, const std::vector<double>& yTrue, 
                  const std::vector<double>& yPred, double lr) 
    {
        // 1. 输出层梯度
        // dLoss/dz2 = yPred - yTrue
        std::vector<double> dLoss_dz2(yPred.size());
        for (size_t o = 0; o < yPred.size(); o++) {
            dLoss_dz2[o] = yPred[o] - yTrue[o];
        }

        // 2. 隐藏层梯度
        // dLoss/dz1 = sum(dLoss/dz2[o] * w2[h][o]) * sigmoid'(z1[h])
        std::vector<double> dLoss_dz1(cached_a1_.size(), 0.0);
        for (size_t h = 0; h < cached_a1_.size(); h++) {
            double sum = 0.0;
            for (size_t o = 0; o < yPred.size(); o++) {
                sum += dLoss_dz2[o] * weights2_[h][o];
            }
            double dSig = SimpleMath::sigmoidDerivative(cached_z1_[h]);
            dLoss_dz1[h] = sum * dSig;
        }

        // 3. 更新 W2, b2
        for (size_t h = 0; h < cached_a1_.size(); h++) {
            for (size_t o = 0; o < yPred.size(); o++) {
                double grad = dLoss_dz2[o] * cached_a1_[h];
                weights2_[h][o] -= lr * grad;
            }
        }
        for (size_t o = 0; o < yPred.size(); o++) {
            bias2_[o] -= lr * dLoss_dz2[o];
        }

        // 4. 更新 W1, b1
        for (size_t i = 0; i < x.size(); i++) {
            for (size_t h = 0; h < cached_a1_.size(); h++) {
                double grad = dLoss_dz1[h] * x[i];
                weights1_[i][h] -= lr * grad;
            }
        }
        for (size_t h = 0; h < cached_a1_.size(); h++) {
            bias1_[h] -= lr * dLoss_dz1[h];
        }
    }

private:
    // 随机初始化层参数
    void initLayer(std::vector<std::vector<double>>& W, 
                   std::vector<double>& B,
                   size_t inDim, size_t outDim)
    {
        W.resize(inDim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-0.1, 0.1); // 增大学习能力

        for (size_t i = 0; i < inDim; i++) {
            W[i].resize(outDim);
            for (size_t j = 0; j < outDim; j++) {
                W[i][j] = dist(gen);
            }
        }

        B.resize(outDim);
        for (size_t j = 0; j < outDim; j++) {
            B[j] = dist(gen);
        }
    }

private:
    // 第一层参数
    std::vector<std::vector<double>> weights1_;
    std::vector<double> bias1_;

    // 第二层参数
    std::vector<std::vector<double>> weights2_;
    std::vector<double> bias2_;

    // 缓存前向传播的中间结果
    std::vector<double> cached_z1_;
    std::vector<double> cached_a1_;
    std::vector<double> cached_z2_;
    std::vector<double> cached_a2_;
};

// 训练函数
void train_mlp(MLP& mlp, 
               const std::vector<std::vector<double>>& X,
               const std::vector<std::vector<double>>& y,
               int epochs,
               double lr) 
{
    for (int e = 0; e < epochs; e++) {
        double total_loss = 0.0;
        
        // 遍历所有样本
        for (size_t i = 0; i < X.size(); i++) {
            std::vector<double> pred = mlp.forward(X[i]);
            total_loss += SimpleMath::binaryCrossEntropy(y[i], pred);

            mlp.backward(X[i], y[i], pred, lr);
        }

        // 每10个epoch打印一次loss
        if (e % 10 == 0) {
            double avg_loss = total_loss / X.size();
            std::cout << "Epoch " << e << ", Average Loss: " << avg_loss << std::endl;
        }
    }
}

/******************************************************************************
 * 第四部分：main 函数 - 演示多层感知器
 *****************************************************************************/
int main()
{
    std::cout << "=== MLP 分类示例 ===\n";

    // 1. 生成训练数据
    DataGenerator dg(100000);  // 生成1000个样本
    std::vector<std::vector<double>> X, y;
    dg.generate_circle_data(X, y);

    // 2. 创建并训练模型
    MLP mlp(2, 8, 1);  // 输入维度=2, 隐藏层=8个神经元, 输出维度=1
    
    std::cout << "开始训练...\n";
    train_mlp(mlp, X, y, 200, 0.01);  // 训练200个epoch，学习率0.01

    // 3. 测试模型准确率
    int correct = 0;
    for (size_t i = 0; i < X.size(); i++) {
        auto pred = mlp.forward(X[i]);
        int predicted_label = pred[0] >= 0.5 ? 1 : 0;
        int true_label = static_cast<int>(y[i][0]);
        if (predicted_label == true_label) {
            correct++;
        }
    }
    double accuracy = static_cast<double>(correct) / X.size() * 100.0;
    std::cout << "训练集准确率: " << accuracy << "%" << std::endl;

    // 4. 测试一些样本点
    std::cout << "\n测试一些样本点：\n";
    for (int i = 0; i < 5; i++) {
        auto pred = mlp.forward(X[i]);
        int predicted_label = pred[0] >= 0.5 ? 1 : 0;
        std::cout << "输入: (" << X[i][0] << ", " << X[i][1] 
                  << "), 预测: " << predicted_label 
                  << ", 实际: " << static_cast<int>(y[i][0]) << std::endl;
    }

    std::cout << "=== MLP 分类示例 结束 ===\n";
    return 0;
}
