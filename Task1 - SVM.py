# -*- coding: utf-8 -*-
"""
Status: COMPLETE

Created on Fri Oct 18 13:54 2019

SMO Algorithm provided by Kristoffer Wickstrøm for use in UiT, FYS-3012

FYS - 3002: Pattern Recognition
Take-Home Exam

Task 1a - Support Vector Machines - Spam/Ham
-------------------------------------------
Support vectors are those data points that can be used to define the optimum "margin"
in between data classes in a classification task. A vector of weights "alphas" as 
well as a threshold value "b" must be generated to determine which vectors will be
used to generate the margins between classes. The smo algorithm handles this task.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist
import scipy.io as sio
import random
import warnings

#Import Training Data
xtr = sio.loadmat('Xtr_spam.mat')
ytr = sio.loadmat('Ytr_spam.mat')
x_train = xtr['Xtr_spam'].T
y_train = ytr['ytr_spam'].T

#Import Evaluation Data
xte = sio.loadmat('Xte_spam.mat')
yte = sio.loadmat('Yte_spam.mat')
x_test = xte['Xte_spam'].T
y_test = yte['yte_spam'].T

"""
Gaussian Kernel:
----------------
The Gaussian kernel is computed between each datapoint xi and each vector in each class of the training data. 
The data is not linearly separable, so the kernel trick is used to represent the classification task in the
Hilbert space via a weighted sum of inner products of vectors. 
"""
def Gauss(x, y, h = 4.5):
    density = np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * h ** 2))
    return density

"""
Classification:
---------------
The decision_function weights the y labels for the support vectors, and finds the dot product with the kernel
matrix of the training data and test data sets against the threshold value b.
"""
# Decision function
def decision_function(alphas, y = y_train[:,0], kernel = Gauss, x_train = x_train, x_test = x_test, b = 0.0):
    result = (alphas * y) @ kernel(x_train, x_test) - b
    return result


"""
Simplified SMO implementation:
------------------------------
Optimizes the SVM optimization problem using a simplified
Sequential Minimal Optimization algorithm. The implementation
is based on [1]. 

Note that the hyperplane is on the form f(x) = w'x + b.

Args:
    K: Numpy matrix of pairwise inner products in the training set or kernel matrix.
        The shape of the matrix should be (N, N), where N is the number of data
        points in the training set.
    y: Numpy vector of labels with values in {-1, 1}. The shape of the vector
        should be (N,)
    C: Weight for the slack variables. A large C forces the margins to be more narrow.
    tol: Tolerance for updating Lagrange multipliers.
    conv_iter: Number of iterations without parameter updates before determining that the
        algorithm has converged.
    max_iter: The maximum number of iterations before the algorithm terminates.

Returns:
    alpha: Lagrange multipliers.
    b: Bias

[1] http://cs229.stanford.edu/materials/smo.pdf

"""
def smo(K, y, C = 1.7, tol = 1e-6, conv_iter = 1000, max_iter = 30000):
    # Ensure that the inputs are numpy arrays
    y = np.asarray(y).flatten()
    K = np.asarray(K)

    N = len(y)

    if not K.shape == (N, N):
        raise ValueError('K should be an %d times %d matrix.' % (N, N))

    # Initialization
    alpha = np.zeros((N,), dtype=float)
    b = 0

    n_iter_conv = 0
    n_iter = 0

    while n_iter_conv < conv_iter and n_iter < max_iter:
        n_pairs_changed = 0

        for i in range(N):
            # Update if values are outside of the given tolerance
            E_i = calculate_E(i, K, alpha, y, b)
            if ((y[i]*E_i < -tol and alpha[i] < C) or (y[i]*E_i > tol and alpha[i] > 0)):
                # Select random j different from i
                j = i
                while j == i:
                    j = random.randint(0, N - 1)

                # Calculate lower and upper bounds
                L, H = calculate_L_H(i, j, y, alpha, C)

                if L == H:
                    continue

                eta = calculate_eta(i, j, K)

                if eta >= 0:
                    continue

                # Evaluate
                E_j = calculate_E(j, K, alpha, y, b)

                alpha_i_old = alpha[i].copy()
                alpha_j_old = alpha[j].copy()

                # Update alpha_j
                alpha[j] = update_alpha(alpha[j], y[j], E_i, E_j, eta, H, L)
                
                # No need to update alpha_i
                if abs(alpha[j] - alpha_j_old) < 1e-5:
                    continue

                # Update alpha_i and calculate new bias
                alpha[i] += y[i]*y[j]*(alpha_j_old - alpha[j])

                # Update bias
                b = calculate_b(b, i, j, K, C, y, alpha, E_i, E_j, alpha_i_old, alpha_j_old)

                n_pairs_changed += 1
        
        # Reset if some alpha pairs have changed
        if n_pairs_changed == 0:
            n_iter_conv += 1
        else:
            n_iter_conv = 0

        n_iter += 1

    if n_iter == max_iter:
        warnings.warn('The SMO algorithm did not converge after %d iterations.' % n_iter, RuntimeWarning)

    return alpha, b

def calculate_E(idx, K, alpha, y, b):
    return ((alpha*y).dot(K[:, idx]) + b - y[idx])

def calculate_L_H(idx_i, idx_j, y, alpha, C):
    y_i = y[idx_i]
    y_j = y[idx_j]
    alpha_i = alpha[idx_i]
    alpha_j = alpha[idx_j]

    if not (y_i == y_j):
        L = max((0, alpha_j - alpha_i))
        H = min((C, C + alpha_j - alpha_i))
    else:
        L = max((0, alpha_i + alpha_j - C))
        H = min((C, alpha_i + alpha_j))

    return (L, H)

def calculate_eta(idx_i, idx_j, K):
    return (2*K[idx_i, idx_j] - K[idx_i, idx_i] - K[idx_j, idx_j])

def update_alpha(alpha_old, y, E_i, E_j, eta, H, L):
    
    # Update alpha
    alpha = alpha_old - y*(E_i - E_j)/eta
    
    # Clamp to valid values
    if alpha > H:
        alpha = H
    elif alpha < L:
        alpha = L
    
    return alpha

def calculate_b(b, idx_i, idx_j, K, C, y, alpha, E_i, E_j, alpha_i_old, alpha_j_old):
    b1 = b - E_i - y[idx_i]*(alpha[idx_i] - alpha_i_old)*K[idx_i, idx_i] - y[idx_j]*(alpha[idx_j] - alpha_j_old)*K[idx_i, idx_j]
    b2 = b - E_j - y[idx_i]*(alpha[idx_i] - alpha_i_old)*K[idx_i, idx_j] - y[idx_j]*(alpha[idx_j] - alpha_j_old)*K[idx_j, idx_j]

    # Ensure that KKT are not violated
    if 0 < alpha[idx_i] < C:
        b = b1
    elif 0 < alpha[idx_j] < C:
        b = b2
    else:
        b = (b1 + b2)/2

    return b

#Prune Training Set to Increase Speed
"""
Prune:
------
Perhaps we can speed up the training process for SVM by eliminating unlikely support vector candidates.
Pairwise Euclidean distances are calculated and sorted; a new training set is developed based on 
those datapoints that are a different classification compared to at least one of k nearest neighbours
(in this example, k == 10). This should reduce the size of the dataset (and increase the speed of 
convergence for the smo algorithm) without greatly impacting the accuracy of the classifier.
"""
def Prune(xtr = x_train, ytr = y_train[:,0]):
    remainder = 0
    new_xtr = []
    new_ytr = []
    distances = squareform(pdist(xtr, metric='euclidean'))
    for i in range(len(xtr[:,0])):
        sort_di = np.argsort(distances[i,:])
        sort_yi = ytr[sort_di][0:10]
        if sort_yi[0] == sort_yi[1] == sort_yi[2] == sort_yi[3] == sort_yi[4] == sort_yi[5] == sort_yi[6] == sort_yi[7] == sort_yi[8] == sort_yi[9]:
            continue
        else:
            new_xtr.append(xtr[i,:])
            new_ytr.append(ytr[i])
            remainder += 1
    return new_xtr, new_ytr

"""
Accuracy:
---------
A confusion matrix is generated to determine accuracy of the entire dataset, as well
as accuracy/misclassification rates for each class (spam or not spam).
"""
def Acc_SVM(y_hat, y):
    true_pos = 0
    false_pos = 0 
    true_neg = 0 
    false_neg = 0
    
    for i in range(len(y_hat)):
        if y_hat[i] >= 0:
            if y[i] == 1:
                true_pos += 1
            else:
                false_pos += 1
        else:
            if y[i] == 1:
                false_neg += 1
            else:
                true_neg += 1
        
    print("Correctly Identified Spam: " + str(true_pos))
    print("Incorrectly Labelled as Spam: " + str(false_pos))
    print("Incorrectly Labelled as Not Spam: " + str(false_neg))
    print("Correctly Identified Normal Emails: " + str(true_neg))
    
    num_correct = true_pos + true_neg
    acc = round(num_correct/(len(y)) * 100, 2)
    
    correct_statement = "Correctly Classified: {0} of {1} observations"
    acc_statement = "Classifier Accuracy: {0} %"
    false_statement = "False Positive Rate: {0} %, False Negative Rate: {1} %"
    print(correct_statement.format(num_correct, len(y)))
    print(acc_statement.format(acc))
    print(false_statement.format(round((false_pos/len(y)) * 100, 2), round((false_neg/len(y)) * 100, 2)))

"""
SVM Classifier:
---------------
The functions are run in order to prepare the training set (Prune), determine the kernel matrix (Gauss),
compute the alphas and b (smo), and decide upon the classification for each data point (decision_function).
The accuracy is presented (Acc_SVM) for assessment of the classifier.
"""
def SVM_Classifier():
    xtr, ytr = Prune()
    xtr = np.array(xtr)
    
    kernel_matrix = Gauss(xtr, xtr, h = 1.6)
    alphas, b = smo(kernel_matrix, ytr)
    
    y_hat = decision_function(alphas = alphas, y = ytr, kernel = Gauss, x_train = xtr, b = b)
    Acc_SVM(y_hat, y_test)
    return None

"""
Suggested Run Script
"""
SVM_Classifier()