from typing import List

import numpy as np
import math
from math import pow

def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:

    assert len(y_true) == len(y_pred)
    tot = 0;
    for (y1,y2) in zip(y_true, y_pred):
        curr = y1 - y2
        curr = pow(curr, 2)
        tot += curr

    tot = tot/len(y_true)
    return tot

    return sum([ pow((x-y), 2) for (x,y) in zip(y_true, y_pred) ])/len(y_pred)
    """raise NotImplementedError"""


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)

    tp = 0
    fp = 0
    fn = 0

    for (x,y) in zip(real_labels, predicted_labels):
        if(x == 1):
            if(x == y):
                tp += 1
            else:
                fp += 1
        else:
            if(y == 1):
                fn += 1

    percision = tp/(tp + fp)
    recall = tp/(tp + fn)
    if(percision + recall == 0):
        return 0
    return 2*percision*recall/(percision + recall)


def polynomial_features(features: List[List[float]], k: int) -> List[List[float]]:
    x = np.matrix(features)
    for i in range(0,k-1):
        x = np.append(x, np.power(np.matrix(features),i+2), 1)
    return x


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    tot = 0;
    for (x,y) in zip(point1, point2):
        z = pow(x-y,2)
        tot += z;
    tot = pow(tot, .5)
    return tot

def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return sum([x*y for x,y in zip(point1, point2)])


def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    tot = euclidean_distance(point1, point2)
    tot = tot*(-.5)
    tot = -math.exp(tot)
    return tot


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        return_list_list = []
        for x in features:
            return_list = []
            distance = inner_product_distance(x,x)
            distance = pow(distance, .5)
            if(distance == 0):
                return features
            for y in x:
                y = y/distance
                return_list.append(y)
            return_list_list.append(return_list)
        return return_list_list



class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """
    
    def __init__(self):
        self.first = False
        self.min_list = None
        self.max_list = None
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        if(not self.first):
            self.first = True
            min_list = []
            max_list = []
            for x in features[:1]:
                for y in x:
                    min_list.append(y)
                    max_list.append(y)
            for x in features:
                for i in range(0,len(x)):
                    if(x[i] < min_list[i]):
                        min_list[i] = x[i]
                    if(x[i] > max_list[i]):
                        max_list[i] = x[i]
            self.min_list = min_list
            self.max_list = max_list

        return_list_list = []
        for x in features:
            return_list = []
            for i in range(0,len(x)):
                y = ((x[i]-self.min_list[i])/(self.max_list[i] - self.min_list[i]))
                return_list.append(y)
            return_list_list.append(return_list)
        
        return return_list_list





