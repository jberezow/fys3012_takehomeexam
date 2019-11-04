# -*- coding: utf-8 -*-
"""
Status: COMPLETE

Created on Tues Oct 22 10:34:00 2019

FYS - 3002: Pattern Recognition
Take-Home Exam

Task 2e - Kernel FDA
--------------------
Implement kernel FDA with a kernel function of your choice and use it to classify both data sets. You
have to find suitable parameters for the kernel function. State the parameters and design choices for your
experiment and provide the final classification error. Plot the kernel FDA projections of the data sets
together with the decision boundary.
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
np.random.shuffle(s)
y = y.T[s]
x = x[s]

"""
Gaussian Kernel:
----------------
The parameter "h" was tested iteratively. Multiple values of h between 0 and 1 reliably produce
100% accuracy.
"""
def Gauss(xi,xt, h = 0.75):
    return np.exp(-(np.linalg.norm(xi[:, np.newaxis] - xt[np.newaxis, :], 2, axis=2)**2)/(2*h**2))
        
"""
Polynomial Kernel:
------------------
The parameters "a" and "b" were tested iteratively. Multiple values of b between reliably 
produce 100% accuracy. Omission of parameter a (a = 1) did not affect the outcome significantly.
P was set to 3 based on the shape of the data (prior assumption). Odd p (5, 7) also work.
"""
def Poly(xi, xt, p = 3, a = 1, b = 1):
    xi = np.asarray(xi)
    xt = np.asarray(xt)
    return (((a * xi.dot(xt.T)) + b) ** p)

#KFDA Function
def KFDA(x, y, Kernel):
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

    """
    Scatter Matrices:
    -----------------
    The N matrix and m0, m1 vectors are prepared here. 
    
    N is constructed based on a quadratic form of the class-specific kernel matrices and an nj * nj
    identity matrix minus a full matrix of values 1/nj, where nj denotes class size.
    
    Epsilon is appended to the N matrix until the determinant is zero to ensure positive-
    definiteness and that all eigenvalues are positive.
    
    The alphas are calculated as the dot product of the inverse of N and (m1 - m0).
    """
    epsilon = 0.001
    n0 = len(x_train0[:,0])
    n1 = len(x_train1[:,0])
    n = n0 + n1
    i0 = np.full((n0,n0), 1/(n0))
    i1 = np.full((n1,n1), 1/(n1))
    ident0 = np.diag(np.ones(n0))
    ident1 = np.diag(np.ones(n1))
    k0 = Kernel(x_train, x_train0)
    k1 = Kernel(x_train, x_train1)
    m0 = (np.sum(k0, axis=1)/n0).reshape(n, 1)
    m1 = (np.sum(k1, axis=1)/n1).reshape(n, 1)
    #M = np.dot((m0 - m1), (m0 - m1).T)
    N = np.dot(np.dot(k0, (ident0 - i0)), k0.T) + np.dot(np.dot(k1, (ident1 - i1)), k1.T)
    while(np.linalg.det(N) == 0):
        N = N + np.diag(np.full(n, epsilon))
    alphas = np.dot(np.linalg.inv(N), (m1 - m0)).T
    
    """
    Threshold:
    ----------
    The means for each class are calculated in order to determine the threshold w0 (c).
    The threshold is the average of the two means.
    """
    m0_mean = np.atleast_2d(np.mean(x_train0, axis=0))
    m1_mean = np.atleast_2d(np.mean(x_train1, axis=0))
    kernal_sum0 = np.dot(alphas, Kernel(x_train, m0_mean))
    kernal_sum1 = np.dot(alphas, Kernel(x_train, m1_mean))
    threshold_c = 0.5*(kernal_sum0 + kernal_sum1)

    """
    Classification:
    ---------------
    A kernel matrix is produced between the training and test data sets. The sum across
    the dot product of the weight vector and each column of the matrix K2 represents the 
    kernel approximation of the transformation of each test vector "xi" in the transformed
    space. The weight is compared against the threshold_c and classified accordingly.
    """
    K2 = Kernel(x_train, x_test)
    weights = np.empty([len(x_test), 1])
    y_hat = []
    
    for j in range(len(x_test[:,0])):
        training_vector = np.atleast_2d(K2[:,j])
        weights[j] = np.dot(alphas, training_vector.T)
        if weights[j] > threshold_c:
            y_hat.append(1)
        else:
            y_hat.append(0)

    """
    Accuracy:
    ---------
    Accuracy is computed to determine success of the classifier.
    """
    num_correct = 0
    for l in range(len(y_hat)):
        if y_hat[l] == y_test[l]:
            num_correct += 1
        else:
            continue
    per_acc = (num_correct/len(y_test))*100
    correct_statement = "Correctly Classified: {0} of {1} observations"
    acc_statement = "Classifier Accuracy: {0} %"
    print("Kernel Fisher Discriminant Analysis with " + str(Kernel.__name__) + " kernel")
    print(correct_statement.format(num_correct, len(y_test)))
    print(acc_statement.format(per_acc))

    """
    Plotting:
    ---------
    A scatter plot is prepared of the test data points coloured accordinging to their reported
    classification. The decision line is produced as a single contour to represent the boundary
    between classification spaces.
    """
    colors = ['green', 'blue']
    fig = plt.figure(figsize=(13,13))
    ax0 = fig.add_subplot(111)
    ax0.scatter(x_test[:,0],x_test[:,1], c=y_hat, cmap=matplotlib.colors.ListedColormap(colors), marker='.', linewidths=3)
            
    x1min = np.min(x_train[:,0])
    x1max = np.max(x_train[:,0])
    x1margin = 0.05*(x1max-x1min)
        
    x2min = np.min(x_train[:,1])
    x2max = np.max(x_train[:,1])
    x2margin = 0.05*(x2max-x2min)
        
    x1axis = np.linspace(x1min - x1margin, x1max + x1margin, 200)
    x2axis = np.linspace(x2min - x2margin, x2max + x2margin, 200)
        
    x1grid = np.tile(x1axis, (len(x2axis),1)).T.reshape(-1, 1)
    x2grid = np.tile(x2axis, (1, len(x1axis))).reshape(-1, 1)
        
    xgrid = np.concatenate((x1grid, x2grid), axis=1)
    xgrid_class = Kernel(x_train, xgrid)
    gridweights = np.empty([len(xgrid), 1])
    y_grid = np.zeros((len(xgrid),1))
    for j in range(len(xgrid[:,0])):
        training_vector = np.atleast_2d(xgrid_class[:,j])
        gridweights[j] = np.dot(alphas, training_vector.T)
        if gridweights[j] > threshold_c:
            y_grid[j] = 1
        else:
            y_grid[j] = 0
            
    x1grid = x1grid.reshape(len(x2axis), -1)
    x2grid = x2grid.reshape(-1, len(x1axis))
    xgrid_class = y_grid.reshape(len(x2axis), len(x1axis))
    ax0.contour(x1grid, x2grid, xgrid_class, levels=[0], linestyles=('dashed'), linewidths=2, colors='k')
    
"""
Suggested Run Script
--------------------
"""
KFDA(x, y, Kernel = Gauss)
KFDA(x, y, Kernel = Poly)