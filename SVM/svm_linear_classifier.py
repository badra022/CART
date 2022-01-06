import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.utils import shuffle

class SVMClassifier:
    
    def __init__(self, kernel = 'linear', n_iterations = 10000, C = 1.0, eta = 1.0):
        self._kernel = kernel
        self._n_iter = n_iterations
        self._C = C
        self._eta = eta
    
    
    def _cost(self, W):
        dist = 1.0 - self._y * np.dot(self._X, W)
        dist[dist < 0] = 0
        
        return (1/2) * np.dot(W, W) + self._C * (1/self._m) * np.sum(dist)
        
    def _get_gradients(self, W, yi, Xi):
        dist = 1.0 - yi * np.dot(Xi, W)
        gradients = np.zeros(len(W))
        if max(0, dist) == 0:
            gradients += W
        else:
            gradients += W - self._C * yi * Xi
            
        gradients = gradients * (1 / self._m)
        return gradients
        
    def fit(self, X, y):
        self._X = np.c_[X, np.ones(X.shape[0])]
        self._y = np.where(y == 0, -1, y)
        self._m = y.size
        self._n = self._X.shape[1]
        W = np.zeros(self._n)
        
        for epoch in range(1, self._n_iter):
#             print("cost = ", self._cost(W), "      (", epoch, ")")
            self._X, self._y = shuffle(self._X, self._y)
            
            for idx, Xi in enumerate(self._X):
                gradients = self._get_gradients(W, self._y[idx], Xi)
                W -= self._eta * gradients
        
        self._W = W
            
            
    def predict(self, X):
        predictions = np.array([self._predict(Xi) for idx, Xi in enumerate(X)])
        return predictions.astype('int')
    
    def _predict(self, Xi):
        return np.dot(self._W[:-1], Xi) + self._W[-1] >= 0
        
    def visualize_2D(self, X_test, y_test):
        plt.scatter(X_test[:,0],X_test[:,1],c = y_test)
        plt.title("Actual")
        plt.show()
        linspace_X0 = np.linspace(X_test[:,0].min(), X_test[:,0].max(), 10)
        linspace_X1 = -1 * (self._W[0] * linspace_X0 + self._W[-1]) / self._W[1]
        plt.plot(linspace_X0, linspace_X1)
        plt.scatter(X_test[:,0],X_test[:,1],c = self.predict(X_test))
        plt.title("Prediction")
        plt.show()