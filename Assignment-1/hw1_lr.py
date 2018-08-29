from __future__ import division, print_function

from typing import List




############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################
import numpy as np
import scipy
from numpy.linalg import inv

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features
        self.weights = None

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        x = np.matrix(features)
        ones = np.ones((len(features), 1))
        x = np.append(ones, x, 1)
        x_trans = x.transpose()
        y = np.matrix(values)
        y = y.transpose()
        c = np.matmul(x_trans,x)
        c_inv = np.linalg.pinv(c)
        x_trans_y = np.matmul(x_trans, y)
        w = np.matmul(c_inv, x_trans_y)
        self.weights = w

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        w = self.weights
        w_trans = w.transpose()
        x = np.matrix(features)
        ones = np.ones((len(features), 1))
        x = np.append(ones, x, 1)
        x = x.transpose()
        new_y = np.matmul(w_trans, x)
        new_y = new_y.tolist()[0]
        return new_y

    def get_weights(self) -> List[float]:
        w = self.weights
        w_list = w.tolist()
        return w_list


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        x = np.matrix(features)
        ones = np.ones((len(features), 1))
        x = np.append(ones, x, 1)
        x_trans = x.transpose()
        y = np.matrix(values)
        y = y.transpose()
        c = np.matmul(x_trans,x)
        identity = np.identity(c.shape[0])
        identity = self.alpha * identity
        c = c+identity
        c_inv = np.linalg.pinv(c)
        x_trans_y = np.matmul(x_trans, y)
        w = np.matmul(c_inv, x_trans_y)
        self.weights = w

    def predict(self, features: List[List[float]]) -> List[float]:
        w = self.weights
        w_trans = w.transpose()
        x = np.matrix(features)
        ones = np.ones((len(features), 1))
        x = np.append(ones, x, 1)
        x = x.transpose()
        new_y = np.matmul(w_trans, x)
        new_y = new_y.tolist()[0]
        return new_y


    def get_weights(self) -> List[float]:
        w = self.weights
        w_list = w.tolist()
        return w_list


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
