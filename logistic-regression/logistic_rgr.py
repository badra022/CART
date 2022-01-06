from math import *
import numpy as np

class LogisticRegression:
    
    def __init__(self, n_iterations = 1000, eta = 0.01):
        self._n_iterations = n_iterations
        self._eta = eta
    
    def _sigmoid(self, h):
        return 1 / (1 + exp(-h))
    
    def _linear_h(self, x, theta):
        return theta.dot(x)
        
    def _cost(self, h, y):
        return -1 * (y * log(h) + (1 - y) * log(1- h))
    
    def _J_theta(self, theta):
        result = 0.0
        for i in range(0, self._m):
            result += self._cost(self._sigmoid(self._linear_h(self._X[i, :], theta)), self._y[i, 0])
        
        return (1 / self._m) * result
        
    def _J_theta__partial_theta(self, theta):
        result = 0.0
        for i in range(0, self._m):
            result += (self._y[i, 0] - self._sigmoid(self._linear_h(self._X[i, :], theta))) * self._X[i, :]
        
        return result
        
    def fit(self, X, y):
        self._m = y.size     #rows
        self._X = X
        self._y = y
        self._n = X.shape[1]   #columns
        
        theta = np.array([0.0] * self._X.shape[1])
        for iter in range(self._n_iterations):
                print("cost: ", self._J_theta(theta))
                theta_prev = theta
                theta = theta + self._eta * self._J_theta__partial_theta(theta)
                if self._J_theta(theta) == self._J_theta(theta_prev):      # convergence case
                    print("Converged to Cost: ", self._J_theta(theta_prev), "\ntheta: ", theta_prev)
                    break      # No need to continue pursueing local Minimum 
                
        self._theta = theta
        
    def predict(self, X):
        return np.array([self._predict(X[i, :]) for i in range(X.shape[0])])
    
    def _predict(self, Xi):
        return int(self._theta.dot(Xi) >= 0.5)