import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

train_X = np.array([[0,0,1],
                    [1,1,1],
                    [1,0,1],
                    [0,1,1]])

train_y = np.array([[0,1,1,0]]).T

np.random.seed(1)

# generate 3 random weights as the number of X features is 3
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Random starting synaptic weights:\n", synaptic_weights)

y_estimate = np.zeros((4,1))

input_layer = train_X

for iter in range(20000):
    y_estimate = sigmoid(np.dot(input_layer, synaptic_weights))

    err = train_y - y_estimate

    adjustments = err * sigmoid_derivative(y_estimate)

    synaptic_weights += np.dot(input_layer.T ,adjustments)

    # if iter % 1000 == 0:
    #     print(f"output in {iter}th iteration:\n", y_estimate)

print("synaptic weights after training:\n", synaptic_weights)
print("outputs after training:\n", y_estimate)