from __future__ import division, print_function

import numpy as np
import scipy as sp
import random
import math

from matplotlib import pyplot as plt
from matplotlib import cm


#######################################################################
# DO NOT MODIFY THE CODE BELOW 
#######################################################################

def binary_train(X, y, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - step_size: step size (learning rate)

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic regression
    - b: scalar, which is the bias of logistic regression

    Find the optimal parameters w and b for inputs X and y.
    Use the average of the gradients for all training examples to
    update parameters.
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D+1)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    i = 0

    ones = np.ones((N,1))
    X = np.append(ones,X,1)


    while(i < max_iterations):
        x_update = np.zeros(D+1)
        for row_index in range(0,N):
            x_curr = X[row_index,0:]
            x_trans = x_curr.transpose()
            wx = np.matmul(w, x_trans)
            a = b + wx.item(0)
            sig = sigmoid(a)
            e = sig - y[row_index]
            x_curr = e*x_curr
            x_update = x_curr + x_update
        x_update = x_update/N
        x_update = step_size*x_update
        x_update = -1*x_update
        w = w + x_update
        i += 1


    """
    TODO: add your code here
    """

    b = w[0]
    w = w[1:]
    assert w.shape == (D,)
    return w, b


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    preds = np.zeros(N) 
    for row_index in range(0,N):
            x_curr = X[row_index,0:]
            x_trans = x_curr.transpose()
            wx = np.matmul(w, x_trans)
            a = b + wx.item(0)
            sig = sigmoid(a)
            if(sig > .5):
                preds[row_index] = 1

    """
    TODO: add your code here
    """   

    assert preds.shape == (N,) 
    return preds

def binary_continuous_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    preds = np.zeros(N) 
    for row_index in range(0,N):
        x_curr = X[row_index,0:]
        x_trans = x_curr.transpose()
        wx = np.matmul(w, x_trans)
        a = b + wx.item(0)
        sig = sigmoid(a)
        preds[row_index] = sig

    """
    TODO: add your code here
    """   

    assert preds.shape == (N,) 
    return preds


def multinomial_train(X, y, C, 
                     w0=None, 
                     b0=None, 
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - step_size: step size (learning rate)
    - max_iterations: maximum number for iterations to perform

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes

    Implement a multinomial logistic regression for multiclass 
    classification. Keep in mind, that for this task you may need a 
    special (one-hot) representation of classification labels, where 
    each label y_i is represented as a row of zeros with a single 1 in
    the column, that corresponds to the class y_i belongs to. 
    """

    N, D = X.shape

    w = np.zeros((C, D + 1))
    if w0 is not None:
        w = w0
    b = np.zeros(C)
    if b0 is not None:
        b = b0


    ones = np.ones((N,1))
    X = np.append(ones,X,1)

    for a in range(N):
        x_1 = X[a,0:]
        x_max = np.argmax(x_1)
        x_max = x_1[x_max]
        for b in range(len(x_1)):
            x_1[b] = x_1[b] - x_max
        X[a,0:] = x_1


    i = 0
    while(i < max_iterations):
        xx_update = np.zeros(w.shape)
        for a in range(N):
            y_hot = np.zeros(C)
            curr = y[a]
            y_hot[curr] = 1
            denominator = 0
            x_update = np.zeros(D+1)
            nums = []
            for j in range(C):
                w_1 = w[j]
                x_1 = X[a, 0:]
                x_trans = np.transpose(x_1)
                wx = np.matmul(w_1, x_trans)
                e_raised = math.exp(wx.item(0))
                denominator += e_raised
                nums.append(e_raised)
            k = 0
            for j in range (len(y_hot)):
                p = nums[k]/denominator
                if(y_hot[j] == 1):
                    x_update = (p-1)*X[a, 0:]
                else:
                    x_update = (p)*X[a, 0:]
                xx_update[j,:] = xx_update[j,:] + x_update
                k += 1
        xx_update = xx_update/N
        xx_update = xx_update*step_size
        w = w - xx_update
        i += 1


    w1 = np.zeros((C,D))
    for i in range(C):
        b = w[:,0]
        w1[i,0:] = w[i,1:]

    w = w1
    """
    TODO: add your code here
    """

    assert w.shape == (C, D)
    assert b.shape == (C,)
    return w, b


def multinomial_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier
    - b: bias terms of the trained multinomial classifier
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes

    Make predictions for multinomial classifier.
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N) 
    

    zeros = np.zeros((b.shape[0],1))
    for i in range(b.shape[0]):
        zeros[i] = b[i]


    w = np.append(zeros, w, 1)


    ones = np.ones((N,1))
    X = np.append(ones,X,1)

    for a in range(N):
        x_1 = X[a,0:]
        x_max = np.argmax(x_1)
        x_max = x_1[x_max]
        for b in range(len(x_1)):
            x_1[b] = x_1[b] - x_max
        X[a,0:] = x_1

    for i in range(N):
        denominator = 0
        nums = []
        for j in range(C):
            w_1 = w[j]
            x_1 = X[i, 0:]
            x_trans = np.transpose(x_1)
            wx = np.matmul(w_1, x_trans)
            e_raised = math.exp(wx.item(0))
            denominator += e_raised
            nums.append(e_raised)
        a = np.matrix(nums)
        a_max = np.argmax(nums)
        preds[i] = a_max




    """
    TODO: add your code here
    """   

    assert preds.shape == (N,)
    return preds


def OVR_train(X, y, C, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array, 
    indicating the labels of each training point
    - C: number of classes in the data
    - w0: initial value of weight matrix
    - b0: initial value of bias term
    - step_size: step size (learning rate)
    - max_iterations: maximum number of iterations for gradient descent

    Returns:
    - w: a C-by-D weight matrix of OVR logistic regression
    - b: bias vector of length C

    Implement multiclass classification using binary classifier and 
    one-versus-rest strategy. Recall, that the OVR classifier is 
    trained by training C different classifiers. 
    """
    N, D = X.shape
    
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    """
    TODO: add your code here
    """
    for i in range(0,C):
        l = []
        for n in y:
            if(n == i):
                l.append(1)
            else:
                l.append(0)
        y_array = np.asarray(l)
        w_curr, b_curr = binary_train(X, y_array)
        w[i,0:] = w_curr[0:]
        b[i] = b_curr



    assert w.shape == (C, D), 'wrong shape of weights matrix'
    assert b.shape == (C,), 'wrong shape of bias terms vector'
    return w, b


def OVR_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained OVR model
    - b: bias terms of the trained OVR model
    
    Returns:
    - preds: vector of class label predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes.

    Make predictions using OVR strategy and predictions from binary
    classifier. 
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N) 
    
    l = np.zeros((C,N))
    for i in range(0, C):
        w_curr = w[i,0:]
        b_curr = b[i]
        a = binary_continuous_predict(X,w_curr,b_curr)
        l[i,0:] = a

    preds = np.argmax(l, axis = 0)


    """
    TODO: add your code here
    """


    assert preds.shape == (N,)
    return preds


#######################################################################
# DO NOT MODIFY THE CODE BELOW 
#######################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def accuracy_score(true, preds):
    return np.sum(true == preds).astype(float) / len(true)

def run_binary():
    from data_loader import toy_data_binary, \
                            data_loader_mnist 

    print('Performing binary classification on synthetic data')
    X_train, X_test, y_train, y_test = toy_data_binary()
        
    w, b = binary_train(X_train, y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))
    
    print('Performing binary classification on binarized MNIST')
    X_train, X_test, y_train, y_test = data_loader_mnist()

    binarized_y_train = [0 if yi < 5 else 1 for yi in y_train] 
    binarized_y_test = [0 if yi < 5 else 1 for yi in y_test] 
    
    w, b = binary_train(X_train, binarized_y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(binarized_y_train, train_preds),
             accuracy_score(binarized_y_test, preds)))

def run_multiclass():
    from data_loader import toy_data_multiclass_3_classes_non_separable, \
                            toy_data_multiclass_5_classes, \
                            data_loader_mnist 
    
    datasets = [(toy_data_multiclass_3_classes_non_separable(), 
                        'Synthetic data', 3), 
                (toy_data_multiclass_5_classes(), 'Synthetic data', 5), 
                (data_loader_mnist(), 'MNIST', 10)]

    for data, name, num_classes in datasets:
        print('%s: %d class classification' % (name, num_classes))
        X_train, X_test, y_train, y_test = data
        
        print('One-versus-rest:')
        w, b = OVR_train(X_train, y_train, C=num_classes)
        train_preds = OVR_predict(X_train, w=w, b=b)
        preds = OVR_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))
    
        print('Multinomial:')
        w, b = multinomial_train(X_train, y_train, C=num_classes)
        train_preds = multinomial_predict(X_train, w=w, b=b)
        preds = multinomial_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))


if __name__ == '__main__':
    
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.output:
            sys.stdout = open(args.output, 'w')

    if not args.type or args.type == 'binary':
        run_binary()

    if not args.type or args.type == 'multiclass':
        run_multiclass()
        