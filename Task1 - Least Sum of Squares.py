# -*- coding: utf-8 -*-
"""
Status: COMPLETE

Created on Wed Oct 16 19:27 2019

FYS - 3002: Pattern Recognition
Take-Home Exam

Task 1a - Least Sum of Squares Classifier - Spam/Ham
----------------------------------------------------
A weighting vector w is generated as a function of xxt and rxy, where xxt is the
matrix as a sum of inner products of training vectors xi, and rxy is the matrix
sum of training vectors xi multiplied by their class value (-1, 1). The optimisation
of the sum of error squares formula leads to an expression concerning the matrix
product of these two matrix - (xxt)^-1 %*% rxy.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import scipy.io as sio

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
Classification:
----------------
The training data set is run through this function to generate the estimate of
the weight vector w_hat. Only several lines of code are required to generate the 
matrices xxt and rxy as described in the classifier description above.
"""
def train_LS(x, y):
    xxt = np.dot(x.T,x)
    xxt_inv = np.linalg.inv(xxt)
    rxy = np.dot(x.T,y)
    w_hat = np.dot(xxt_inv, rxy)
    return w_hat

"""
Test data is classified as the inner product of the weight vector w and the test
vector x.
"""
def test_LS(x, w):
    return np.dot(w.T, x.T).T

"""
Accuracy:
---------
A confusion matrix is generated to determine accuracy of the entire dataset, as well
as accuracy/misclassification rates for each class (spam or not spam).
"""
def Acc_LS(y_hat, y):
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
Suggested Run Script
"""
def LS_Classifier(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test):
    w_hat = train_LS(x_train, y_train)
    y_hat = test_LS(x_test, w_hat)
    Acc_LS(y_hat, y_test)
    return None

LS_Classifier(x_train, y_train, x_test, y_test)