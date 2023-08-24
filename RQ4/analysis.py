import pickle
from math import sqrt,ceil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from scipy.special import logsumexp, softmax
import os



'''
KL divergence
'''

def mean_kl(data1, data2):
  """Compute mean KL divergence (across all examples) between 2 data distributions."""
  log_softmax1 = data1 - logsumexp(data1, axis=-1, keepdims=True)
  log_softmax2 = data2 - logsumexp(data2, axis=-1, keepdims=True)
  kl = np.sum(np.exp(log_softmax1) * (log_softmax1 - log_softmax2), -1)
  return np.mean(kl)

kl_mean = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.AUTO)

'''
normal cka
'''
def centering_c(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(H,K)
#linear kernel -----------------------------------------------------be using
def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.
  Args:
    x: A num_examples x num_features matrix of features.
  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)
#centerd gram -------------------------------------------------be using
def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.
  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.
  center the X and Y matrix features and then find the covariance matrix
  for the kernel calculation,
  you can center the feature first, or perform the function operation after the kernel calculation,
  which is equivalent to centering the features of X and Y before the kernel calculation(linear kernel),the variance can be equivalent to a linear kernel
    Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.可能为负
  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T):
    
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)# column
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram

#CKA --------------------------------------------------------be using
def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.
  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.
  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels抵消 for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel()) 
  #<vec(X,X.T),vec(Y,Y.T)> 等价于 Tr(X,X.T,Y,Y.T)  等价于||Y.T,X||2/F
  normalization_x = np.linalg.norm(gram_x)  #||X.T,X||2/F = HSIC(K,K)
  normalization_y = np.linalg.norm(gram_y)  #||Y.T,Y||2/F = HSIC(L,L)

  return scaled_hsic / (normalization_x * normalization_y)

#Choose Linear Kernel
def cka_linear(X,Y,debiased = False):
    if debiased:
        return cka(gram_linear(X), gram_linear(Y),True)
    else:
        return cka(gram_linear(X), gram_linear(Y))

