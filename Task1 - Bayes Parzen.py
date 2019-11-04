# -*- coding: utf-8 -*-
"""
Status: COMPLETE

Created on Thurs Oct 17 20:30 2019

FYS - 3002: Pattern Recognition
Take-Home Exam

Task 1a - Bayes with Parzen Windowing - Spam/Ham
------------------------------------------------
The probability that an email is "Spam" given a weight vector wi, by Bayes' rule, is proportional to 
the joint probability of occurences of the components of the weight vector multipled by the probability
of incurring a spam email. 

We use an estimate of s based on the proportion of the training set that is spam,and calculate the 
probability of experiencing a given value for each componenent of wi using the Parzen windowing method 
(Kernel Density Estimation) with a Gaussian kernel. 

Under the Naive Bayes framework, the components of the weight vector are assumed to be iid, meaning the
probability estimation for the entire wi vector is equal to the product of probabilities.
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
Gaussian Kernel:
----------------
The Gaussian kernel is computed between each datapoint xi and each vector in each class of the training data. 
This returns a posterior distribution estimation that is necessary for the classification task.
"""
def Gauss(xi, x, h = 0.8):
    density = np.exp(-np.dot((x - xi).T, (x - xi))/((2*h)**2))
    return density

"""
Density Estimation:
----------------
The Parzen function organizes the test data to obtain posterior density estimations as a sum of kernel
values. Both posteriors are returned for use in the classification function.
"""
def Parzen(xi, x_train, y_train, h = 0.8):
    x_spam = x_train[y_train[:,0] == 1]
    x_not = x_train[y_train[:,0] == -1]
    posterior_s = 0
    posterior_ns = 0
    for x in x_spam:
        #kernel = (xi - x)/h
        posterior_s += Gauss(xi, x)
    for x in x_not:
        #kernel = (xi - x)/h
        posterior_ns += Gauss(xi, x)
    return posterior_s/(len(x_spam)), posterior_ns/len(x_not)

"""
Classification:
----------------
Using Bayes' rule, the product of the prior and the posterior return two values - the comparison of
which can be used to determine for which class each test data point should be labelled. The prior
for each class is taken to be the proportion of the dataset that appears as spam. The posterior
is calculated using Parzen windows with a Gaussian kernel.
"""
def Bayes(x_train, x_test, y_train, y_test):
    prior_s = len([1 for j in y_train if j == 1]) / len(y_train)
    prior_ns = len([1 for j in y_train if j == -1]) / len(y_train)
    y_hat = []
    for i in x_test:
        posterior_s, posterior_ns = Parzen(i, x_train, y_train)
        if (posterior_s * prior_s) > (posterior_ns * prior_ns):
            y_hat.append(1)
        else:
            y_hat.append(-1)
    return np.atleast_2d(y_hat).T

"""
Accuracy:
---------
A confusion matrix is generated to determine accuracy of the entire dataset, as well
as accuracy/misclassification rates for each class (spam or not spam).
"""
def Acc_BC(y_hat, y):
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
def Bayes_Classifier(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test):
    y_hat = Bayes(x_train, x_test, y_train, y_test)
    Acc_BC(y_hat, y_test)
    return None

Bayes_Classifier(x_train, y_train, x_test, y_test)