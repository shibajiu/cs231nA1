import numpy as np
from matplotlib import pyplot as plt
import random
import time
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.classifiers.linear_svm import svm_loss_vectorized
from cs231n.classifiers import LinearSVM

# Load the raw CIFAR-10 data.
cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We will also make a development set, which is a small subset of
# the training set.
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# As a sanity check, print out the shapes of the data
print('Training data shape: ', X_train.shape)
print('Validation data shape: ', X_val.shape)
print('Test data shape: ', X_test.shape)
print('dev data shape: ', X_dev.shape)

mean_img=np.mean(X_train,axis=0)
print(mean_img[:10]) # print a few of the elements
# plt.figure(figsize=(4,4))
# plt.imshow(mean_img.reshape((32,32,3)).astype('uint8'))
# plt.show()

X_train -= mean_img
X_val -= mean_img
X_test -= mean_img
X_dev -= mean_img

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)

# generate a random SVM weight matrix of small numbers
# W = np.random.randn(3073, 10) * 0.0001

# loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.000005)
# print('loss: %f' % (loss, ))
#
# # Compute the loss and its gradient at W.
# loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)
#
# # Numerically compute the gradient along several randomly chosen dimensions, and
# # compare them with your analytically computed gradient. The numbers should match
# # almost exactly along all dimensions.
# from cs231n.gradient_check import grad_check_sparse
# f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
# grad_numerical = grad_check_sparse(f, W, grad)
#
# # do the gradient check once again with regularization turned on
# # you didn't forget the regularization gradient did you?
# loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
# f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]
# grad_numerical = grad_check_sparse(f, W, grad)

# Next implement the function svm_loss_vectorized; for now only compute the loss;
# we will implement the gradient in a moment.

# # Complete the implementation of svm_loss_vectorized, and compute the gradient
# # of the loss function in a vectorized way.
#
# # The naive implementation and the vectorized implementation should match, but
# # the vectorized version should still be much faster.
# tic = time.time()
# _, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
# toc = time.time()
# print('Naive loss and gradient: computed in %fs' % (toc - tic))
#
# tic = time.time()
# _, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
# toc = time.time()
# print('Vectorized loss and gradient: computed in %fs' % (toc - tic))
#
# # The loss is a single number, so it is easy to compare the values computed
# # by the two implementations. The gradient on the other hand is a matrix, so
# # we use the Frobenius norm to compare them.
# difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
# print('difference: %f' % difference)

tsvm = LinearSVM()

# tic = time.time()
# loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
#                       num_iters=1500, verbose=True)
# toc = time.time()
# print('That took %fs' % (toc - tic))

#
# # A useful debugging strategy is to plot the loss as a function of
# # iteration number:
# plt.plot(loss_hist)
# plt.xlabel('Iteration number')
# plt.ylabel('Loss value')
# plt.show()

# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
# y_train_pred = svm.predict(X_train)
# print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
# y_val_pred = svm.predict(X_val)
# print('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))

learning_rates = [1e-7, 5e-5]
regularization_strengths = [2.5e4, 5e4]

results = {}
best_val = -1  # The highest validation accuracy that we have seen so far.
best_svm = None  # The LinearSVM object that achieved the highest validation rate.

################################################################################
# TODO:                                                                        #
# Write code that chooses the best hyperparameters by tuning on the validation #
# set. For each combination of hyperparameters, train a linear SVM on the      #
# training set, compute its accuracy on the training and validation sets, and  #
# store these numbers in the results dictionary. In addition, store the best   #
# validation accuracy in best_val and the LinearSVM object that achieves this  #
# accuracy in best_svm.                                                        #
#                                                                              #
# Hint: You should use a small value for num_iters as you develop your         #
# validation code so that the SVMs don't take much time to train; once you are #
# confident that your validation code works, you should rerun the validation   #
# code with a larger value for num_iters.                                      #
################################################################################
tic = time.time()
for ilr in np.arange(learning_rates[0],learning_rates[1],0.05*(learning_rates[1]-learning_rates[0])):
    for ireg in np.arange(regularization_strengths[0],regularization_strengths[1],0.05*(regularization_strengths[1]-regularization_strengths[0])):
        svm = LinearSVM()
        _ = svm.train(X_train, y_train, learning_rate=ilr, reg=ireg,
                              num_iters=1500, verbose=True)
        y_train_pred = svm.predict(X_train)
        train_acc=np.mean(y_train == y_train_pred)
        print('training accuracy: %f' % (train_acc))
        y_val_pred = svm.predict(X_val)
        val_acc=np.mean(y_val==y_val_pred)
        if val_acc>best_val:
            best_val=val_acc
            best_svm=svm
        print('validation accuracy: %f' % (val_acc))
        results[(ilr,ireg)]=(train_acc,val_acc)

################################################################################
#                              END OF YOUR CODE                                #
################################################################################

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)
toc = time.time()
print('That took %fs' % (toc - tic))
import winsound
winsound.Beep(600,1000)
import math
x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# plot training accuracy
marker_size = 100
colors = [results[x][0] for x in results]
plt.subplot(2, 1, 1)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')

# plot validation accuracy
colors = [results[x][1] for x in results] # default size of markers is 20
plt.subplot(2, 1, 2)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')
plt.show()

y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)

# Visualize the learned weights for each class.
# Depending on your choice of learning rate and regularization strength, these may
# or may not be nice to look at.
w = best_svm.W[:-1, :]  # strip out the bias
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)

    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])
