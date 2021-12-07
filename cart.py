import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from imblearn.over_sampling import SMOTE
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from sklearn.model_selection import train_test_split
import math

class Node:
    def __init__(self, accr, num_samples, num_samples_per_cls, predicted_cls):
        """
        parameters:
            accr: accuracy metric (gini or Entropy)
            num_samples : total number of samples in this Node
            num_samples_per_cls: num of samples per cls -> array of numbers (one for each cls)
            predicted_cls: predicted cls of this Node
            
        return: 
            Node obj.
        """
        self.accr = accr
        self.num_samples = num_samples
        self.num_samples_per_cls = num_samples_per_cls
        self.predicted_cls = predicted_cls
        self.left_child = None          # link to the left child if exist
        self.right_child = None         # link to the right child if exist
        self.feature_idx = 0           # which feature we split the dataset with 
        self.threshold = 0             # which feature value we split the dataset with


class CART:
    def __init__(self, max_depth=None, metric = "gini"):
        self.max_depth = max_depth
        self.metric = metric
        
    def _gini(self, cls_col):
        return 1.0 - (sum(sum(cls_col == cls) ** 2 for cls in set(cls_col)) / (len(cls_col) ** 2))
    
    def _entropy(self, cls_col):
        p_plus = sum(cls_col == 1) / cls_col.size
        p_minus = sum(cls_col == 0) / cls_col.size
        try:
            return -1 * (p_plus * math.log2(p_plus) + p_minus * math.log2(p_minus))
        except:
            return 0.0

    def _gain(self, parent_entropy, num_of_classes_left, num_of_classes_right, thr_idx, num_samples):
        p0_left = num_of_classes_left[0] / thr_idx
        p1_left = num_of_classes_left[1] / thr_idx
        p0_right = num_of_classes_right[0] / (num_samples - thr_idx)
        p1_right = num_of_classes_right[1] / (num_samples - thr_idx)
        
        try:
            entropy_of_left_child = -1.0 * (p0_left * math.log2(p0_left) + p1_left * math.log2(p1_left))
        except:
            entropy_of_left_child = 0.0
        try:
            entropy_of_right_child = -1.0 * (p0_right * math.log2(p0_right) + p1_right * math.log2(p1_right))
        except:
            entropy_of_right_child = 0.0
        
        return parent_entropy - ( (thr_idx / num_samples) * entropy_of_left_child +
                                  ((num_samples - thr_idx) / num_samples) * entropy_of_right_child)
    
    def _avg_childs_gini(self, num_of_classes_left, num_of_classes_right, thr_idx, num_samples):
        """return the average impurity of the two children, weighted by their population"""
        
        gini_left = 1.0 - sum((num_of_classes_left[x] / thr_idx) ** 2 for x in range(self.n_classes_))
        gini_right = 1.0 - sum((num_of_classes_right[x] / (num_samples - thr_idx)) ** 2 for x in range(self.n_classes_))

        # The Gini impurity of a split is the weighted average of the Gini
        # impurity of the children.
        return (thr_idx * gini_left + (num_samples - thr_idx) * gini_right) / num_samples

    def fit(self, X, y):
        """Build decision tree classifier."""
        self.n_classes_ = len(set(y.reshape(-1)))         # indicies - 0 -> n-1
        self.n_features_ = X.shape[1]
        self.tree_ = self._extend_tree(X, y)
        """ _extend_tree() function is a recursive function that build all the nodes and leafs of the tree
        and return the root node of the tree, this node is connected with it's right and left childs
        and each node is connected to it's childs and so on """
        
    def _best_split(self, X, y):
        """Find the best split for a node.
        "Best" means that the average impurity of the two children, weighted by their
        population, is the smallest possible. Additionally it must be less than the
        impurity of the current node.

        best_idx: Index of the feature for best split, or None if no split is found.
        best_thr: Threshold value to use for the split, or None if no split is found.
        """
        # Need at least two elements to split a node.
        m = y.size
        if m <= 1:
            return None, None

        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))       # Sort data along selected feature.

            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):  # possible split positions
                c = int(classes[i - 1])
                num_left[c] += 1
                num_right[c] -= 1

                if thresholds[i] == thresholds[i - 1]:
                    continue

                gini = self._avg_childs_gini(num_left, num_right, i, m)
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint                    

        return best_idx, best_thr
    
    def _best_split_entropy_version(self, X, y):
        """Find the best split for a node.
        "Best" means that the average gain of the two children, weighted by their
        population, is the largest possible.

        best_idx: Index of the feature for best split, or None if no split is found.
        best_thr: Threshold value to use for the split, or None if no split is found.
        """
        # Need at least two elements to split a node.
        m = y.size
        if m <= 1:
            return None, None

        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        best_entropy = self._entropy(y)
        parent_entropy = best_entropy
        best_entropy = 0
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))       # Sort data along selected feature.

            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):  # possible split positions
                c = int(classes[i - 1])
                num_left[c] += 1
                num_right[c] -= 1

                if thresholds[i] == thresholds[i - 1]:
                    continue

                gain = self._gain(parent_entropy, num_left, num_right, i, m)
                if gain > best_entropy:
                    best_entropy = gain
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint                    

        return best_idx, best_thr
    
    def _extend_tree(self, X, y, depth=0):
        """Build a decision tree by recursively finding the best split."""
        # Population for each class in current node. The predicted class is the one with
        # largest population.
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = int(np.argmax(num_samples_per_class))
#         print("predicted_classes: ", predicted_class)
        if self.metric == "gini":
            node = Node(
                accr=self._gini(y.reshape(-1)),
                num_samples=y.size,
                num_samples_per_cls=num_samples_per_class,
                predicted_cls=predicted_class,
            )
        elif self.metric == "entropy":
            node = Node(
                accr=self._entropy(y.reshape(-1)),
                num_samples=y.size,
                num_samples_per_cls=num_samples_per_class,
                predicted_cls=predicted_class,
            )

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth:
            idx, thr = 0, 0
            if self.metric == "gini":
                idx, thr = self._best_split(X, y)
            elif self.metric == "entropy":
                idx, thr = self._best_split_entropy_version(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_idx = idx
                node.threshold = thr
#                 print("node:\n gini: {} \n num_samples: {} \n threshold: {} \n feature_idx: {} \n predicted class: {} \n ------------\n".format(node.accr,
#                                                                                                                          node.num_samples,
#                                                                                                                          node.threshold,
#                                                                                                                          node.feature_idx,
#                                                                                                                          node.predicted_cls))
                node.left_child = self._extend_tree(X_left, y_left, depth + 1)
                node.right_child = self._extend_tree(X_right, y_right, depth + 1)
        return node
    
    def predict(self, X):
        """Predict class for a group of samples."""
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree_
        while node.left_child:
#             print("node: \nnum_samples:",node.num_samples, "\ngini:", node.accr, "\n")
            if inputs[node.feature_idx] < node.threshold:
                node = node.left_child
            else:
                node = node.right_child
        return node.predicted_cls