from __future__ import print_function

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers import KNearestNeighbor
import matplotlib.pyplot as plt
import time


def timecost(f, *args):
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic


def cross_validation(X_train, Y_train):
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = []
    y_train_folds = []
    ################################################################################
    # TODO:                                                                        #
    # Split up the training data into folds. After splitting, X_train_folds and    #
    # y_train_folds should each be lists of length num_folds, where                #
    # y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
    # Hint: Look up the numpy array_split function.                                #
    ################################################################################
    X_train_folds = np.array(np.split(X_train, num_folds))
    y_train_folds = np.array(np.split(Y_train, num_folds))
    i_num_class=X_train.shape[1]
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################

    # A dictionary holding the accuracies for different values of k that we find
    # when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the different
    # accuracy values that we found when using that value of k.
    k_to_accuracies = {}

    ################################################################################
    # TODO:                                                                        #
    # Perform k-fold cross validation to find the best value of k. For each        #
    # possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
    # where in each case you use all but one of the folds as training data and the #
    # last fold as a validation set. Store the accuracies for all fold and all     #
    # values of k in the k_to_accuracies dictionary.                               #
    ################################################################################
    for i in k_choices:
        k_to_accuracies[i] = []
    for i in range(num_folds):
        i_train = np.append(X_train_folds[:i],X_train_folds[i + 1:],axis=0).reshape(-1,X_train.shape[1])
        iy_train = np.append(y_train_folds[:i],y_train_folds[i + 1:]).flatten()
        i_validation = X_train_folds[i]
        iy_validation = y_train_folds[i]
        classifier.train(i_train, iy_train)
        i_dist = classifier.compute_distances_no_loops(i_validation)
        for kc in k_choices:
            i_pred = classifier.predict_labels(i_dist, kc)
            i_num_correct=np.sum(i_pred==iy_validation)
            accuracy=i_num_correct/i_num_class
            print('At k:%d, Got %d / %d correct => accuracy: %f' % (kc, i_num_correct, i_num_class, accuracy))
            k_to_accuracies[kc] = np.append(k_to_accuracies[kc],accuracy)

    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################

    # Print out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))

    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'

# try:
#     del X_train,Y_trian,X_test,Y_test
#     print('previous data cache cleared')
# except:
#     pass

X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', Y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxs=np.flatnonzero(Y_train==y)
#     idxs=np.random.choice(idxs,samples_per_class,False)
#     for i,id in enumerate(idxs):
#         plt_num=i+y*samples_per_class+1
#         plt.subplot(samples_per_class,num_classes,plt_num)
#         plt.imshow(X_train[id].astype('uint8'))
#         if i == 0:
#             plt.title(cls)
# plt.show()

# subsample the data for more efficient code execution in this exercise
num_train = 5000
X_train = X_train[range(num_train)]
Y_train = Y_train[range(num_train)]
num_test = 500
X_test = X_test[range(num_test)]
Y_test = Y_test[range(num_test)]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, Y_train)
dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)
# plt.imshow(dists, interpolation='none')
# plt.show()

# y_test_pred = classifier.predict_labels(dists, k=1)
# num_correct=np.sum(y_test_pred==Y_test,axis=0)
# accuracy = float(num_correct) / num_test
# print('At k:%d, Got %d / %d correct => accuracy: %f' % (1, num_correct, num_test, accuracy))
#
# y_test_pred = classifier.predict_labels(dists, k=5)
# num_correct = np.sum(y_test_pred == Y_test)
# accuracy = float(num_correct) / num_test
# print('At k:%d, Got %d / %d correct => accuracy: %f' % (5, num_correct, num_test, accuracy))
#
# dists_one = classifier.compute_distances_one_loop(X_test)
# difference=np.linalg.norm(dists-dists_one,ord='fro')
# print('Difference was: %f' % (difference, ))
# if difference < 0.001:
#     print('Good! The distance matrices are the same')
# else:
#     print('Uh-oh! The distance matrices are different')

# dists_two = classifier.compute_distances_no_loops(X_test)
# # check that the distance matrix agrees with the one we computed before:
# difference = np.linalg.norm(dists - dists_two, ord='fro')
# print('Difference was: %f' % (difference, ))
# if difference < 0.001:
#     print('Good! The distance matrices are the same')
# else:
#     print('Uh-oh! The distance matrices are different')

# two_loop = timecost(classifier.compute_distances_two_loops, X_test)
# print('Two loop version took %f seconds' % two_loop)
#
# one_loop = timecost(classifier.compute_distances_one_loop, X_test)
# print('One loop version took %f seconds' % one_loop)
#
# no_loop = timecost(classifier.compute_distances_no_loops, X_test)
# print('No loop version took %f seconds' % no_loop)

# cross_validation(X_train,Y_train)

best_k = 10

classifier = KNearestNeighbor()
classifier.train(X_train,Y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == Y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
