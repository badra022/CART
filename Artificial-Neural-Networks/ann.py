import numpy as np
import random
import sys

class ANN():

    def __init__(self, list_layers):
        # initialize a list of weights, each for one layer, with it's shape (layer's neurons num x input's neurons num)
        self.weights = [np.random.random_sample(size=(list_layers[idx],list_layers[idx - 1])) for idx in range(1, len(list_layers))]
        self.biases = [np.random.random_sample(size=(list_layers[idx], 1)) for idx in range(1, len(list_layers))]

        # for w in self.weights:
        #     print(w.shape)

        # for b in self.biases:
        #     print(b.shape)


        # initializing some useful attributes
        self.num_layers = len(list_layers)
        self.input_shape = (list_layers[0], 1)
        # print("=====\n", self.input_shape)
        self.output_shape = (list_layers[-1], 1)
        # print(self.output_shape)

    def _sigmoid(self, z):
        """ calculate the sigmoid function of a numpy array (z) """
        return 1 / (1 + np.exp(-z))

    def _d_sigmoid(self, z):
        """ calculate the derivative sigmoid function of a numpy array (z) """
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def feed_forward(self, input):
        """
        input: (1, num_of_features) shaped numpy array
        ===========================================================
        Description: claculating the output of the Network in case we feed it some passed (input)   
                     using forwardPropagation
        """

        # modifying the shape of the input to fit in the input layer
        activation = input.reshape(self.input_shape)

        # looping over each layer's (weights, biases) and calculate the resulting activation of the layer
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, activation) + bias
            # print(f"bias shape: {bias.shape}")
            activation = self._sigmoid(z)
            # print(f"a shape: {activation.shape}")

        # return the final estimated output of the Network reshaped to (1, num_of_features)
        return activation

    def back_propagate(self, xsample, ysample):
        """
        xsample: (1, num_of_features) shaped numpy array
        ysample: integer label 0 or 1
        =========================================================
        return: (delta_weights, delta_biases) that is required to modify the actual (weights, biases) of the Network
                to make the Network predict the estimated output of that particular xsample correctly and more accurate
        """
        # One-Hot encoding for the sample class (depending on the output layer neurons num the user entered)
        self.outputs = np.zeros(self.output_shape)
        self.outputs[ysample, 0] = 1

        # for o in self.outputs:
        #     print(o, end=", ")
        # print(f"[{ysample}]")
        
        # pre-init
        d_loss_d_a = 0
        activations = []
        z = []
        delta_weights = self.weights.copy()
        delta_biases = self.biases.copy()

        # forwardPropagation Method to fill out activations list and z list to be used in the backpropagation process
        curr_activation = xsample.reshape(self.input_shape)
        activations.append(curr_activation)

        # looping over each layer's (weights, biases) and calculate the resulting activation of the layer
        for weight, bias in zip(self.weights, self.biases):
            # calculating Z and adding it to the list
            curr_z = np.dot(weight, curr_activation) + bias
            z.append(curr_z)
            # print(f"z: {curr_z.shape}")
            # print(f"bias: {bias.shape}")
            # print(f"weight: {weight.shape}")
            # print(f"a: {curr_activation.shape}")

            # calculating activation and adding it to the list
            curr_activation = self._sigmoid(curr_z)
            activations.append(curr_activation)

        # assigning the last activation to final estimated_outputs
        estimated_outputs = activations[-1]

        # calculating delta_weights and delta_biases of the Last layer, using Chain Rule of calculus
        # (refer to some reference to know how)
        d_loss_d_a = 2 * self.d_loss_d_a(self.outputs, estimated_outputs)
        # print(self.outputs)
        # print(estimated_outputs)
        # print(z[-1].shape)
        d = np.multiply(d_loss_d_a, self._d_sigmoid(z[-1]))
        delta_weights[-1] = np.dot(d, activations[-2].T)
        delta_biases[-1] = d.copy()

        # looping over all the Layers (in reversed order) and calculate each layer's (delta_weights, delta_biases)
        for L in range(2, self.num_layers):
            d = np.multiply(self._d_sigmoid(z[-L]), self.weights[-L+1].T.dot(d))
            delta_weights[-L] = np.dot(d, activations[-L-1].T)
            delta_biases[-L] = d.copy()

        # return the result
        return delta_weights, delta_biases

    def SGD(self, x_mini_batch, y_mini_batch, lr, m):
        """
        x_mini_batch    : (num_mini_batch, num_features) shaped numpy array
        y_mini_batch    : (num_mini_batch, 1) shaped numpy array
        lr              : learning rate used for updating (weights, biases)
        ======================================================================
        Description     : this function perform the Mini Batch Stochastic gradient descent, 
                          calculating delta (weights, biases) resulted from backpropagating the Network using
                          each sample (x, y) in the mini batch, adding each of these delta (weights, biases) to
                          one resulting Delta (weights biases) for actual updating the (weights, biases).
        """
        # the resulting Delta (weights, biases) which will be used to update the (weights, biases) of the Network
        Delta_weights = [np.zeros(self.weights[i].shape) for i in range(self.num_layers-1)]
        Delta_biases = [np.zeros(self.biases[i].shape) for i in range(self.num_layers-1)]
        # print(Delta_weights[0])
        # for w in Delta_weights:
        #     print(w.shape)

        # for b in Delta_biases:
        #     print(b.shape)

        # Loop for each sample (x, y) in the selected mini batch
        for xsample, ysample in zip(x_mini_batch, y_mini_batch):
            
            """get the delta (weights, biases) that is required for the Network to update the (weights, biases) with,
            to be able to correctly predict this specific sample (x, y),
            adding this delta to the resulting Delta (weights, biases), which will be used after iterating all
            the samples in the mini batch to update the (weights, biases)"""
            delta_weights, delta_biases = self.back_propagate(xsample, ysample)

            # adding the delta (weights biases) to one big resulting Delta (weights, biases)

            # for D, d in zip(Delta_weights, delta_weights):
            #     print(D.shape, d.shape)
            # print(Delta_weights[0])
            Delta_weights = [D + d for D, d in zip(Delta_weights, delta_weights)]
            Delta_biases = [D + d for D, d in zip(Delta_biases, delta_biases)]
        # sys.exit()
        # updating the weights and biases using the Delta (weights, biases)
        # print(Delta_weights[-1])
        self.weights = [weight - (lr * Delta_weight) for weight, Delta_weight in zip(self.weights, Delta_weights) ]
        # print(self.weights[0])
        self.bias = [bias - (lr * Delta_bias) for bias, Delta_bias in zip(self.biases, Delta_biases) ]

    def fit(self, x, y, epochs=5, lr=0.1, mini_batch_size=20):
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
        # attaching y with x (for shuffling)
        examples = np.hstack((x, y))

        # loop over epochs
        for epoch in range(epochs):

            # shuffle the training dataset (x, y)
            random.shuffle(examples)

            # splitting the training dataset to mini batches for x and y apart
            x_mini_batches = np.vsplit(examples[:, :-1], mini_batch_size)
            y_mini_batches = np.vsplit(examples[:, -1].reshape(-1, 1), mini_batch_size)

            # loop over each mini batch to apply mini batch stochastic gradient descent
            for x_mini_batch, y_mini_batch in zip(x_mini_batches, y_mini_batches):

                #apply the mini batch stochastic GD for (x, y) mini batch with learning rate (lr)
                self.SGD(x_mini_batch, y_mini_batch, lr, len(x_mini_batch))
                
                # print the status (training_accuracy) of the current epoch
                # print(f"epoch {epoch+1}/{epochs}: accuracy: {self.evaluate(x, y).round(2)}")

            # print the status (training_accuracy) of the current epoch
            print(f"epoch {epoch+1}/{epochs}: accuracy: {self.evaluate(x, y).round(2)}")

    def loss(self, actual, estimated):
        """ calculate the loss by MSE"""
        return (actual - estimated) ** 2

    def d_loss_d_a(self, actual, estimated):
        """ calculate the derivative of the loss MSE"""
        return actual - estimated

    def evaluate(self, x_test, y_test):
        """
        x_test               : (n, num_features) shaped numpy array
        y_test               : (n, 1) shaped numpy array
        ======================================================
        return: accuracy score on the test (x, y) passed to the function
        """
        # get the predictions of the x_test from the Network
        predictions = self.predict(x_test).reshape(-1)

        # reshape the y_test to be 1D iterable array which will be used in comparison
        y_test = y_test.reshape(-1)

        # calculate and return the accuracy (number of true predictions / number of all predictions)
        num_test_data = len(x_test)
        return (sum(predictions == y_test) / num_test_data) * 100

    def predict(self, x_test):
        """
        x_test: (n, num_features) shaped numpy array
        ==============================================
        return: array of estimated classes for the testing dataset (x_test) passed to the function
        """
        
        # calculate the One Hot encoded Outputs of the Network when feed with x_test
        estimated_y = np.array([self.feed_forward(sample).argmax(axis=0) for sample in x_test]).reshape(-1, 1)

        # calculate the corresponding class of the one hot encoded array (which is the index number) and return it
        return estimated_y.argmax(axis=1)



x = np.random.randint(low=0, high=11, size=(20, 3)) # three features, in 10 rows
y = np.random.randint(low=0, high=3, size=(20, 1))

model = ANN([3, 4, 4, 3])
model.fit(x, y, epochs=5, lr=0.1, mini_batch_size=1)