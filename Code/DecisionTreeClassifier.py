from __future__ import division
import random
import numpy as np
from scipy.stats import mode
from functions import entropy_gain, entropy


class Decision_Tree(object):
 
    def __init__(self, max_features=lambda x: x, max_depth=10,
                    min_split=2):
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_split = min_split

    def Train(self, X, y):
        
        features = X.shape[1]
        sub_features = int(self.max_features(features))
        feature_indices = random.sample(xrange(features), sub_features)
        self.trunk = self.build_tree(X, y, feature_indices, 0)

    def Predict(self, X):
     
        num_samples = X.shape[0]
        y = np.empty(num_samples)
        for j in xrange(num_samples):
            node = self.trunk

            while isinstance(node, Node):
                if X[j][node.feature_index] <= node.threshold:
                    node = node.branch_true
                else:
                    node = node.branch_false
            y[j] = node

        return y

    def build_tree(self, X, y, feature_indices, depth):
       

        if depth is self.max_depth or len(y) < self.min_split or entropy(y) is 0:
            return mode(y)[0][0]
        
        feature_index, threshold = find_split(X, y, feature_indices)

        X_true, y_true, X_false, y_false = split(X, y, feature_index, threshold)
        if y_true.shape[0] is 0 or y_false.shape[0] is 0:
            return mode(y)[0][0]
        
        branch_true = self.build_tree(X_true, y_true, feature_indices, depth + 1)
        branch_false = self.build_tree(X_false, y_false, feature_indices, depth + 1)

        return Node(feature_index, threshold, branch_true, branch_false)


class Node(object):
    
    def __init__(self, feature_index, threshold, branch_true, branch_false):
        self.feature_index = feature_index
        self.threshold = threshold
        self.branch_true = branch_true
        self.branch_false = branch_false


def split(X, y, feature_index, threshold):
    
    X_true = []
    y_true = []
    X_false = []
    y_false = []

    for j in xrange(len(y)):
        if X[j][feature_index] <= threshold:
            X_true.append(X[j])
            y_true.append(y[j])
        else:
            X_false.append(X[j])
            y_false.append(y[j])

    X_true = np.array(X_true)
    y_true = np.array(y_true)
    X_false = np.array(X_false)
    y_false = np.array(y_false)

    return X_true, y_true, X_false, y_false


def find_split(X, y, feature_indices):
    
    num_features = X.shape[1]
    best_gain = 0
    best_feature_index = 0
    best_threshold = 0
    for feature_index in feature_indices:
        values = sorted(set(X[:, feature_index])) 

        for j in xrange(len(values) - 1):
            threshold = (values[j] + values[j+1])/2
            X_true, y_true, X_false, y_false = split(X, y, feature_index, threshold)
            gain = entropy_gain(y, y_true, y_false)

            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold
