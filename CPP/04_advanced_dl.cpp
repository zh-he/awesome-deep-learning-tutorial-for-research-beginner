#include <torch/torch.h>
#include <iostream>
#include <iomanip>  // for std::setw
#include <chrono>

// ---------------------------------
//  定义 LeNet 模型
//  输入：1x28x28
//  结构：
//     Conv1(1->6, kernel=5x5) -> ReLU -> MaxPool(2x2)
//     Conv2(6->16, kernel=5x5) -> ReLU -> MaxPool(2x2)
//     Flatten
//     Linear1(16*4*4 -> 120) -> ReLU
//     Linear2(120 -> 84) -> ReLU
//     Linear3(84 -> 10)
// ---------------------------------
struct LeNet : torch::nn::Module {
    // 定义网络层
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};

    // 构造函数（初始化各层）
    LeNet() {
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1, 6, /*kernel_size=*/5)));
        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(6, 16, /*kernel_size=*/5)));

        fc1 = register_module("fc1", torch::nn::Linear(16 * 4 * 4, 120));
        fc2 = register_module("fc2", torch::nn::Linear(120, 84));
        fc3 = register_module("fc3", torch::nn::Linear(84, 10));
    }

    // 前向传播
    torch::Tensor forward(torch::Tensor x) {
        // input x: [N, 1, 28, 28]
        // 卷积 + ReLU + 2x2 池化
        x = torch::relu(conv1->forward(x));
        x = torch::max_pool2d(x, {2, 2});
        // 第二层卷积 + ReLU + 2x2 池化
        x = torch::relu(conv2->forward(x));
        x = torch::max_pool2d(x, {2, 2});
        // 展平
        x = x.view({x.size(0), -1});
        // 全连接层
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        // 最后一层不激活，直接输出 logits
        x = fc3->forward(x);
        return x;
    }
};

// --------------------------------------------------
//  训练函数（使用模板形参来匹配任意 DataLoader 类型）
// --------------------------------------------------
template <typename DataLoader>
void train(
    size_t epoch,
    LeNet& model,
    torch::optim::Optimizer& optimizer,
    DataLoader& data_loader,   // 模板参数，自动推导
    torch::Device device,
    size_t dataset_size
) {
    model.train();
    size_t batch_index = 0;

    // 遍历每个 batch
    for (auto& batch : data_loader) {
        // 获取数据与标签并拷贝到指定的 device (CPU/GPU)
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device, torch::kLong);

        // 前向传播
        auto output = model.forward(data);
        // 计算损失
        auto loss = torch::nn::functional::cross_entropy(output, targets);

        // 反向传播 & 参数更新
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // 打印训练信息
        if (batch_index % 100 == 0) {
            std::cout << "Train Epoch: " << epoch
                      << " [" << std::setw(5) << (batch_index * batch.data.size(0))
                      << "/" << dataset_size << "] "
                      << "Loss: " << loss.item<float>() << std::endl;
        }

        ++batch_index;
    }
}

// --------------------------------------------------
//  测试函数（同样使用模板形参）
// --------------------------------------------------
template <typename DataLoader>
void test(
    LeNet& model,
    DataLoader& data_loader,  // 模板参数，自动推导
    torch::Device device,
    size_t dataset_size
) {
    torch::NoGradGuard no_grad; // 禁用梯度计算
    model.eval();

    double test_loss = 0.0;
    int32_t correct = 0;

    for (const auto& batch : data_loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device, torch::kLong);

        auto output = model.forward(data);

        // 整个 batch 的交叉熵损失 (reduction = Sum)
        auto loss = torch::nn::functional::cross_entropy(
            output, 
            targets, 
            torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum)
        );

        test_loss += loss.item<double>();

        // 计算分类正确的数量
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().item<int64_t>();
    }

    // 求平均损失
    test_loss /= dataset_size;
    // 准确率
    auto accuracy = static_cast<double>(correct) / dataset_size;

    std::cout << "\nTest set: Average loss: " << test_loss
              << ", Accuracy: " << (accuracy * 100.0) << "%\n" << std::endl;
}

// --------------------------------------------------
//  main 函数
// --------------------------------------------------
int main() {
    // 如果你有可用 GPU，可以改为 torch::kCUDA
    torch::Device device(torch::kCPU);

    // 设置超参数
    const int64_t batch_size = 64;
    const size_t num_epochs = 5;
    const double learning_rate = 0.001;

    // 指定数据集所在路径
    std::string mnist_data_path = "your path";

    try {
        // 构建训练数据集
        auto train_dataset = torch::data::datasets::MNIST(
            mnist_data_path,
            torch::data::datasets::MNIST::Mode::kTrain
        )
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

        // 训练集大小
        auto train_size = train_dataset.size().value();

        // 构建训练 DataLoader (随机采样)
        auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(2)
        );

        // 构建测试数据集
        auto test_dataset = torch::data::datasets::MNIST(
            mnist_data_path,
            torch::data::datasets::MNIST::Mode::kTest
        )
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

        // 测试集大小
        auto test_size = test_dataset.size().value();

        // 构建测试 DataLoader (顺序采样)
        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_dataset),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(2)
        );

        // 创建模型
        LeNet model;
        model.to(device);

        // 创建优化器 (Adam)
        torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(learning_rate));

        // 开始训练与测试
        for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
            auto start = std::chrono::high_resolution_clock::now();

            // 注意: 把 *train_loader / *test_loader 传入模板函数
            train(epoch, model, optimizer, *train_loader, device, train_size);
            test(model, *test_loader, device, test_size);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Epoch " << epoch << " finished in " 
                      << elapsed.count() << " seconds.\n" << std::endl;
        }

    } catch (const c10::Error& e) {
        std::cerr << "Error loading dataset or running the model: " << e.msg() << std::endl;
        return -1;
    }

    std::cout << "Training finished successfully!\n";
    return 0;
}
