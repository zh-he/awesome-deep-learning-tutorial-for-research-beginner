#include <torch/torch.h>
#include <iostream>
#include <iomanip>

//====================//
//    定义 LeNet 模型  //
//====================//

// 经典 LeNet-5 (略微简化), 用于单通道 28x28 MNIST
struct LeNetImpl : public torch::nn::Module {
    // 卷积层 + 激活 + 池化 层
    // conv1: in_channels=1, out_channels=6, kernel_size=5 (LeNet 经典)
    // conv2: in_channels=6, out_channels=16, kernel_size=5
    // 然后全连接层 fc1=120, fc2=84, fc3=10
    // 激活可用 ReLU，也可用传统的 Tanh/Sigmoid
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    LeNetImpl() {
        // 卷积: 1->6, kernel=5
        conv1 = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1, 6, 5).stride(1).padding(0));
        // 卷积: 6->16, kernel=5
        conv2 = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(6, 16, 5).stride(1).padding(0));

        // 全连接: 16 * 4 * 4 -> 120
        // 因为: (28x28) -> conv1(5x5) -> pool(2x2) => (6@12x12)
        //        -> conv2(5x5) -> pool(2x2) => (16@4x4)
        fc1 = torch::nn::Linear(16 * 4 * 4, 120);
        fc2 = torch::nn::Linear(120, 84);
        fc3 = torch::nn::Linear(84, 10);

        // 注册
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("fc1",   fc1);
        register_module("fc2",   fc2);
        register_module("fc3",   fc3);
    }

    torch::Tensor forward(torch::Tensor x) {
        // LeNet 典型流程:
        // 1) conv1 -> ReLU -> avg/max pool(2x2)
        // 2) conv2 -> ReLU -> pool(2x2)
        // 3) flatten -> fc1 -> ReLU -> fc2 -> ReLU -> fc3 -> (后面可接softmax)

        // conv1
        x = conv1->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, 2); // kernel=2, stride=2

        // conv2
        x = conv2->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, 2);

        // flatten
        x = x.view({x.size(0), -1}); // batch_size x (16*4*4)

        // fc1
        x = fc1->forward(x);
        x = torch::relu(x);

        // fc2
        x = fc2->forward(x);
        x = torch::relu(x);

        // fc3
        x = fc3->forward(x);
        // 通常最后需要 LogSoftmax 或者在外部使用 CrossEntropyLoss
        // 这里直接返回 logits
        return x;
    }
};
TORCH_MODULE(LeNet); // 使用宏包装，让我们可以用 LeNet model;

//===================================//
//   读取 MNIST 并进行训练/测试的示例   //
//===================================//
int main() {
    try {
        // 如果有 GPU 且想用:
        //  torch::Device device(torch::kCUDA);
        // 否则用CPU:
        torch::Device device(torch::kCPU);

        // 1. 超参数
        const int64_t batch_size = 64;
        const int64_t num_epochs = 5;
        const double learning_rate = 0.001;

        // 2. 加载 MNIST 数据集
        // 请确保在此路径下存在 MNIST 的:
        // train-images-idx3-ubyte, train-labels-idx1-ubyte,
        // t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte
        std::string mnist_data_path = "./mnist";

        auto train_dataset = torch::data::datasets::MNIST(
                mnist_data_path, 
                torch::data::datasets::MNIST::Mode::kTrain)
                .map(torch::data::transforms::Normalize<>(0.1307, 0.3081)) // 标准化
                .map(torch::data::transforms::Stack<>());
        auto train_size = train_dataset.size().value();

        auto test_dataset = torch::data::datasets::MNIST(
                mnist_data_path, 
                torch::data::datasets::MNIST::Mode::kTest)
                .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                .map(torch::data::transforms::Stack<>());
        auto test_size = test_dataset.size().value();

        // 3. 数据加载器 (DataLoader)
        auto train_loader = torch::data::make_data_loader(
            std::move(train_dataset),
            batch_size
        );
        auto test_loader = torch::data::make_data_loader(
            std::move(test_dataset),
            batch_size
        );

        // 4. 实例化网络 & 优化器
        LeNet model;    // 我们的 LeNet
        model->to(device); // 放到 CPU or GPU

        torch::optim::Adam optimizer(
            model->parameters(), 
            torch::optim::AdamOptions(learning_rate)
        );

        // 5. 训练循环
        for(int epoch = 1; epoch <= num_epochs; epoch++) {
            model->train(); // 训练模式
            double running_loss = 0.0;

            int batch_index = 0;
            for (auto &batch : *train_loader) {
                // 取数据
                auto data = batch.data.to(device);
                auto targets = batch.target.to(device, torch::kLong);

                optimizer.zero_grad();
                auto output = model->forward(data);
                auto loss = torch::nn::functional::cross_entropy(
                                output, targets);

                loss.backward();
                optimizer.step();

                running_loss += loss.item<double>() * data.size(0);

                batch_index++;
            }

            double train_epoch_loss = running_loss / train_size;
            std::cout << "Epoch [" << epoch << "/" << num_epochs 
                      << "], Loss: " << std::fixed << std::setprecision(4) 
                      << train_epoch_loss << std::endl;

            // ============== 测试阶段 ==============
            model->eval();
            torch::NoGradGuard no_grad;
            int correct = 0;
            double test_loss = 0.0;

            for (auto &batch : *test_loader) {
                auto data = batch.data.to(device);
                auto targets = batch.target.to(device, torch::kLong);

                auto output = model->forward(data);
                auto loss = torch::nn::functional::cross_entropy(output, targets);
                test_loss += loss.item<double>() * data.size(0);

                auto prediction = output.argmax(1); // shape: [batch_size]
                correct += prediction.eq(targets).sum().item<int>();
            }

            double avg_loss = test_loss / test_size;
            double accuracy = static_cast<double>(correct) / test_size;

            std::cout << "Test set: Average loss: " << std::setprecision(4) << avg_loss
                      << ", Accuracy: " << correct << "/" << test_size 
                      << " (" << (accuracy * 100.) << "%)\n";
        }

    } catch (const c10::Error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
