import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  print(X.shape, dW.shape)
  # https://cs231n.github.io/linear-classify/#loss-function
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct = y[i]
    for j in range(num_classes):
      if j == correct:
        continue
      margin = scores[j] - scores[correct] + 1 # NOTE: delta = 1
      if margin > 0:
        loss += margin
        dW.T[j] += X[i]
        dW.T[correct] -= X[i]

  # Right now the loss and gradient are a sum over all training examples,
  # but we want them to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization.
  loss += reg * np.sum(W ** 2)
  dW += 2 * reg * W  # NOTE: no sum here because each weight has different value

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = np.empty(W.shape)
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # scores.shape, scores[y].shape = (500, 10), (500, 3073)
  scores = np.matmul(X, W)  # X cross W; [row.dot(W) for row in X]
  # TODO: make single loop
  for i in range(len(X)):
    current = y[i]
    margins = np.maximum(0, scores[i] - scores[i, current] + 1) # NOTE: delta = 1
    margins[current] = 0
    loss[i] = margins

  loss = loss.sum()
  # TypeError: 'numpy.float64' object does not support item assignment
  #loss[loss == 1] = 0  # don't incur delta penalty for correct answer
  loss /= X.shape[0]
  loss += reg * np.sum(W ** 2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
