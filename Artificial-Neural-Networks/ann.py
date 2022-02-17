import numpy as np
import random

class ANN():

    def __init__(self, list_layers):

        # initialize a list of weights, each for one layer, with it's shape (layer's neurons num x input's neurons num)
        self.weights = [np.random.randn((list_layers[idx],list_layers[idx - 1])) for idx in range(1, len(list_layers))]
        self.biases = [np.random.randn((list_layers[idx], 1)) for idx in range(1, len(list_layers))]
        self.num_layers = len(list_layers)
        self.input_shape = (list_layers[0], 1)
        self.output_shape = (list_layers[-1], 1)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _d_sigmoid(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def feed_forward(self, input):
        activation = input.T
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, activation) + bias
            activation = self._sigmoid(z)
        return activation.reshape(-1)

    def back_propagate(self, xsample, ysample):
        # One-Hot encoding for the sample class (depending on the output layer neurons num the user entered)
        self.outputs = np.zeros(self.output_shape)
        self.outputs[ysample, 0] = 1
        d_loss_d_a = 0
        activations = []
        z = []
        delta_weights = self.weights.copy()
        delta_biases = self.biases.copy()

        curr_activation = xsample.T
        for weight, bias in zip(self.weights, self.biases):
            curr_z = np.dot(self.weight, curr_activation) + self.bias
            z.append(curr_z)
            curr_activation = self._sigmoid(curr_z)
            activations.append(curr_activation)
        estimated_outputs = activations[-1]
        d_loss_d_a = 2 * self.d_loss_d_a(self.outputs, estimated_outputs)
        d = d_loss_d_a.dot(self._d_sigmoid(z[-1]))
        delta_weights[-1] = np.dot(d, activations[-2].T)
        delta_biases[-1] = d

        for L in range(2, self.num_layers):
            d = np.dot(self._d_sigmoid(z[-L]), self.weights[-L+1].T.dot(d))
            delta_weights[-L] = np.dot(d, activations[-L].T)
            delta_biases[-L] = d

        return delta_weights, delta_biases

    def SGD(self, x_mini_batch, y_mini_batch, lr):
        Delta_weights = [np.zeros(self.weights[i].shape) for i in range(self.num_layers)]
        Delta_biases = [np.zeros(self.biases[i].shape) for i in range(self.num_layers)]

        for xsample, ysample in zip(x_mini_batch, y_mini_batch):
            delta_weights, delta_biases = self.back_propagate(xsample, ysample)
            Delta_weights = [D + d for D, d in zip(Delta_weights, delta_weights)]
            Delta_biases = [D + d for D, d in zip(Delta_biases, delta_biases)]

        self.weights = [weight - lr * Delta_weight for weight, Delta_weight in zip(self.weights, Delta_weights) ]
        self.bias = [bias - lr * Delta_bias for bias, Delta_bias in zip(self.biases, Delta_biases) ]

    def fit(self, x, y, epochs=5, lr=0.01, mini_batch_size=20):
        """
        x               : (n, num_features) shaped numpy array
        y               : (n, 1) shaped numpy array
        epochs          : numper of training epochs
        lr              : learning rate for graddient descent
        mini_batch_size : parameter for mini batch stochastic gradient descent
        =========================================================================================
        Description     : this function takes the training X and y dataset and configurations,
                          then fit the Neural Network to them using the passed configurations,
                          the optimizer for updating the weights and biases that this function uses is
                          mini batch gradient descent
        """
        self.n_samples = len(x)
        d_loss_d_a = 0
        a = []
        z = []

        # attaching y with x (for shuffling)
        examples = np.hstack((x, y))

        # loop over epochs
        for epoch in range(epochs):

            # shuffle the training dataset (x, y)
            random.shuffle(examples)

            # splitting the training dataset to mini batches for x and y apart
            x_mini_batches = np.vsplit(examples[:, :-1], mini_batch_size)
            y_mini_batches = np.vsplit(examples[:, -1], mini_batch_size)

            # loop over each mini batch to apply mini batch stochastic gradient descent
            for x_mini_batch, y_mini_batch in zip(x_mini_batches, y_mini_batches):

                #apply the mini batch stochastic GD for (x, y) mini batch with learning rate (lr)
                self.SGD(x_mini_batch, y_mini_batch, lr)

            # print the status (training_accuracy) of the current epoch
            print(f"epoch {epoch}/{epochs}: accuracy: {self.evaluate(x, y)}")

    def loss(self, actual, estimated):
        return (actual - estimated) ** 2

    def d_loss_d_a(self, actual, estimated):
        return actual - estimated

    def evaluate(self, x_test, y_test):
        num_test_data = len(x_test)
        predictions = self.predict(x_test).reshape(-1)
        y_test = y_test.reshape(-1)
        return (sum(predictions == y_test) / num_test_data) * 100

    def predict(self, x_test):
        estimated_y = [self.feed_forward(sample) for sample in x_test].to_numpy()
        return estimated_y.argmax(axis=1)







model = ANN([784, 512, 256, 10])