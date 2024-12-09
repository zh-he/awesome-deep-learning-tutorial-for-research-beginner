import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def MSE(y_predict, y_true):
    return np.mean((y_predict - y_true) ** 2) / 2


def MSE_derivative(y_predict, y_true):
    return y_predict - y_true


def binary_cross_entropy(y_predict, y_true):
    epsilon = 1e-10
    return -np.mean(y_true * np.log(y_predict + epsilon) + (1 - y_true) * np.log(1 - y_predict + epsilon))


def bce_derivative(y_predict, y_true):
    epsilon = 1e-10
    return -(y_true / (y_predict + epsilon)) + (1 - y_true) / (1 - y_predict + epsilon)




class MLP(object):
    def __init__(self, input_dim, hidden_dims):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        self.weights = {}
        self.bias = {}

        self.weight_gradients = {}
        self.bias_gradients = {}

        self.activations = {}
        self.u = {}

        self.loss = []
        self.accuracy = []

        for i in range(0, self.num_layers):
            if i == 0:
                self.weights[i] = np.random.randn(self.input_dim, self.hidden_dims[i])
            else:
                self.weights[i] = np.random.randn(self.hidden_dims[i - 1], self.hidden_dims[i])
            self.bias[i] = np.zeros((1, self.hidden_dims[i]))

    def forward(self, x):
        self.x = x
        for i in range(0, self.num_layers):
            if i < self.num_layers - 1:
                self.u[i] = np.dot(x, self.weights[i]) + self.bias[i]
                x = relu(self.u[i])
                self.activations[i] = x
            else:
                self.u[i] = np.dot(x, self.weights[i]) + self.bias[i]
                x = sigmoid(self.u[i])
            self.activations[i] = x
        return self.activations[self.num_layers - 1]

    def backward(self, loss_derivative):
        delta = None
        for i in range(self.num_layers - 1, -1, -1):
            if i == self.num_layers - 1:
                delta = loss_derivative * sigmoid_derivative(self.activations[i])
                self.weight_gradients[i] = np.dot(self.activations[i - 1].T, delta)
            else:
                delta = np.dot(delta, self.weights[i + 1].T) * relu_derivative(self.u[i])
                if i == 0:
                    self.weight_gradients[i] = np.dot(self.x.T, delta)
                else:
                    self.weight_gradients[i] = np.dot(self.activations[i - 1].T, delta)

            self.bias_gradients[i] = delta

    def update(self, batch_size, learning_rate):
        for i in range(0, self.num_layers):
            self.weights[i] -= learning_rate * self.weight_gradients[i] / batch_size
            self.bias[i] -= learning_rate * np.mean(self.bias_gradients[i], axis=0, keepdims=True)

    def train(self, x, y, batch_size, learning_rate, epochs, loss_function, loss_derivative, shuffle=True):
        for i in range(epochs):

            if shuffle:
                data = np.hstack((x, y))
                np.random.shuffle(data)
                x = data[:, :-1]
                y = data[:, -1:]

            # batch_accuracies = []
            # batch_losses = []

            predictions = []

            batch = (len(x) + batch_size - 1) // batch_size
            for j in range(batch):
                x_batch = x[j * batch_size: (j + 1) * batch_size, :]
                y_batch = y[j * batch_size: (j + 1) * batch_size, :]

                output = self.forward(x_batch)
                self.backward(loss_derivative(output, y_batch))
                self.update(batch_size, learning_rate)

                predictions.append(output)
                # batch_losses.append(loss_function(output, y_batch))
                # prediction_classes = np.where(output > 0.5, 1, 0)
                # batch_accuracies.append(np.mean(prediction_classes == y_batch))

            # loss = np.mean(batch_losses)
            # accuracy = np.mean(batch_accuracies)

            predictions = np.vstack(predictions)
            loss = loss_function(predictions, y)
            predicted_classes = np.where(predictions > 0.5, 1, 0)
            accuracy = np.mean(predicted_classes == y)
            print(f'[Epoch {i + 1}/{epochs}] loss: {loss:.4f}, accuracy: {accuracy:.4f}')

            self.loss.append(loss)
            self.accuracy.append(accuracy)


if __name__ == '__main__':
    np.random.seed(42)
    x = np.random.randn(1000, 5)
    y = ((x[:, 0] > 0).astype(int) & (x[:,1] > 0.5).astype(int) & (x[:, 2] < 0)).reshape(-1, 1)

    mlp = MLP(input_dim=x.shape[1],
              hidden_dims=[32, 16, 1])

    mlp.train(x=x,
              y=y,
              epochs=50,
              batch_size=64,
              learning_rate=0.1,
              loss_function=binary_cross_entropy,
              loss_derivative=bce_derivative)

    import matplotlib.pyplot as plt

    plt.plot(mlp.loss, label='loss')
    plt.plot(mlp.accuracy, label='accuracy')
    plt.legend()
    plt.grid()
    plt.show()
