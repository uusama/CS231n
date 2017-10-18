import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_training = X.shape[0]
    num_classes = W.shape[1]
    for i in xrange(num_training):
        s = X[i].dot(W)
        f = s - np.max(s)
        # p = np.exp(f[y[i]]) / np.sum(np.exp(f))  # p =si
        s = lambda k: np.exp(f[k]) / np.sum(np.exp(f))
        s_i = s(y[i])
        loss += - np.log(s_i)
        # for the derivative equation refer to http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression#Cost_Function
        for j in range(num_classes):
            s_j = s(j)
            dW[:, j] += (s_j - (y[i] == j)) * X[i]
    loss /= num_training
    dW /= num_training
    if reg > 0:
        loss += .5 * reg * np.sum(W * W)
        dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_training = X.shape[0]
    f = X.dot(W)  # n ,10
    maxf = np.max(f, axis=1)[:, np.newaxis]
    f = -maxf + f
    p = np.exp(f[np.arange(num_training), y]) / np.sum(np.exp(f), axis=1)
    loss = -np.log(p)
    # # reurns the same loss
    # exp_f = np.exp(f)
    # sum_exp_f = np.log(np.sum(p,axis=1))
    # loss = -f[np.arange(num_training),y] + exp_f

    # # for the derivative equation refer to http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression#Cost_Function
    ind = np.zeros_like(f)
    ind[np.arange(num_training), y] = 1
    p = np.exp(f) / np.sum(np.exp(f), axis=1)[:, np.newaxis]
    dW = X.T.dot(p - ind)
    dW = dW / num_training + reg * W
    loss = np.sum(loss)
    loss = np.sum(loss) / num_training + 0.5 * reg * np.sum(W * W)
    return loss, dW
