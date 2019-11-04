# -*- coding: utf-8 -*-
"""
Status: COMPLETE

Created on Mon Oct 21 14:47:00 2019

FYS - 3002: Pattern Recognition
Take-Home Exam

Task 2a, 2b - FDA w/ PCA Comparison
-----------------------------------
Implement FDA and use it to classify the data set. Split the data set into training data and test data.
State the parameters in your experiment and provide the classification error. Plot the data set together
with the decision boundary in input space.

Compare the direction of the weight vector you obtain with FDA with the direction of the first principal
component from linear PCA, which should also be plotted with the data. Explain the difference.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio

#Load Data
data = sio.loadmat('moons_dataset.mat')
x = data['X']
y = np.array(data['y'])

#Split Into Training and Test Set
s = np.arange(x.shape[0])
np.random.seed(seed=23)
np.random.shuffle(s)
y = y.T[s]
x = x[s]

"""
The dataset is split into training and test data points. Here, the dataset is split 50/50, but the
classifier can be successful with a variety of proportions (75/25, 25/75).
The training data has been shuffled above, so it must be sorted according to each data point's class.
"""
x_train = x[0:199]
x_test = x[200:399]
y_train = y[0:199]
y_test = y[200:399]
x_train0 = np.array([x_train[i] for i in range(len(y_train)) if y_train[i] == 0])
x_train1 = np.array([x_train[i] for i in range(len(y_train)) if y_train[i] == 1])
y_train0 = np.array([y_train[i] for i in range(len(y_train)) if y_train[i] == 0])
y_train1 = np.array([y_train[i] for i in range(len(y_train)) if y_train[i] == 1])

#Mean Vectors
mu0 = np.mean(x_train0,axis=0).reshape(2,) 
mu1 = np.mean(x_train1,axis=0).reshape(2,)
muT = 0.5 * (mu0 + mu1)
mu0b = np.mean(x_train0,axis=0).reshape(2,1) 
mu1b = np.mean(x_train1,axis=0).reshape(2,1)
muTb = np.mean(x_train, axis=0).reshape(2,1)


"""
Scatter matrices
----------------
sw and sb represent the within-class and between-class matrices used to solve the eigenvalue 
problem for the first principal component of the data.
"""
sw0 = np.dot((x_train0 - mu0).T,(x_train0 - mu0))
sw1 = np.dot((x_train1 - mu1).T,(x_train1 - mu1))
sw = 0.5*(sw0 + sw1)
sm = np.dot((x_train - muT).T, (x_train - muT))
sb = 0.5*(np.dot((mu1b - muTb),(mu1b - muTb).T) + np.dot((mu0b - muTb),(mu0b - muTb).T))

#Eigenvalues, Eigenvectors
evalues, evectors = np.linalg.eig(np.dot(np.linalg.inv(sw),sb))
pca1 = [evectors[0][1], evectors[1][1]]

"""
FDA Classification:
------------------
The discriminant function g(x) = (mu1 - mu2).T Sw^-1x + w0
Where w1 is the inner product of the difference in mean vectors between classes, and w0
is determined as the inner product of w1 with the total mean vector muT.
"""
w1 = ((mu0 - mu1).T @ np.linalg.inv(sw)).reshape(2,1)
w0 = np.dot(w1.T, muT)

def FDA(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test, w = w1):
    y = (np.dot(w1.T, x_test.T) - w0).T
    results = np.zeros(len(y))
    for i in range(len(results)):
        if y[i] < 0:
            results[i] = 1
        else:
            results[i] = 0
    PlotFDA(x_test, results)
    Acc_FDA(results, y_test[:,0])

"""
Plotting:
--------
Data points are plotted and coloured according to their classification. The black line
represents the Fisher discriminant line. The purple line represents the first
principal component of the matrix ratio sb/sw.
"""
def PlotFDA(x_test, results):
    colors = ['green', 'blue']
    fig = plt.figure(figsize=(13,13))
    ax0 = fig.add_subplot(111)
    ax0.scatter(x_test[:,0],x_test[:,1], c=results, cmap=matplotlib.colors.ListedColormap(colors), marker='.', linewidths=3)
    slope = w1[0]/(-w1[1])
    intercept = w0 / (-w1[1])
    x_grid = np.linspace(-1.5, 2.5, 1000)
    abline_values = [slope * i - intercept for i in x_grid]
    ax0.plot(x_grid, abline_values, color='black', linestyle='-', linewidth=2, label = "Discriminant Line")
    
    slope2 = pca1[0]/(-pca1[1])
    x_grid = np.linspace(-1.5, 2.5, 1000)
    abline_values2 = [slope2 * i for i in x_grid]
    ax0.plot(x_grid, abline_values2, color='purple', linestyle='-', linewidth=2, label = "First Principal Component")
    ax0.legend(prop={'size': 16})

"""
Accuracy:
--------
The number of correctly classified data points is tallied and compared to the size of
the test dataset. A percentage of correctly classified points is expressed.
"""
def Acc_FDA(y_hat, y):
    num_correct = np.sum(np.array(y_hat) == np.array(y))
    per_acc = round(num_correct/(len(y_test)) * 100, 2)
    correct_statement = "Correctly Classified: {0} of {1} observations"
    acc_statement = "Classifier Accuracy: {0} %"
    print(correct_statement.format(num_correct, len(y)))
    print(acc_statement.format(per_acc))
    
"""
Suggested Run Script
--------------------
"""
FDA()