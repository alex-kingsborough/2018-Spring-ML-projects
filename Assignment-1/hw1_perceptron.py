from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt
import random
import math

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = 2
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        i = 0
        converge = False
        while(i < self.max_iteration):
            i+=1
            check_w = self.w

            for (a,b) in zip(features, labels):
                w = np.matrix(self.w)
                x = np.matrix(a)
                x = x.transpose()
                new_y = np.matmul(w, x)
                new_y = new_y.tolist()[0]
                y_sign = -1
                if(new_y[0] > 0):
                    y_sign = 1
                if(y_sign != b):
                    tot = 0
                    for l in x:
                        tot += l**2
                    tot = math.sqrt(tot)
                    w = w + ((b*x).transpose()/tot)
                    self.w = w.tolist()[0]
            if(self.w == check_w):
                converge = True
                break

        return converge



    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        return_list=[]
        w = np.matrix(self.w)
        x = np.matrix(features)
        x = x.transpose()
        new_y = np.matmul(w, x)
        new_y = new_y.tolist()[0]
        for a in new_y:
            if(a > 0):
                return_list.append(1)
            else:
                return_list.append(-1)
        return return_list



        
        raise NotImplementedError

    def get_weights(self) -> Tuple[List[float], float]:
        return self.w
    