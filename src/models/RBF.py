"""ELE2761 Exercise 4: Policy approximation.

Implements the provided functionality to be used in your solution.

CLASSES
    DDPG         -- Deep Deterministic Policy Gradient network
    Memory       -- Replay Memory

FUNCTIONS
    rbfprojector -- Gaussian RBF projector factory.
"""

from math import pi
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import tensorflow as tf
import gym

def __gaussrbf(s, p, v, sigma):
    """Gaussian radial basis function activation.
    
       f = gaussrbf(s, p, v, sigma) returns the activation for the
       radial basis functions specified by (`p`, `v`, `sigma`) calculated at
       `s`. `p` is a list of position centers, `v` is a list of velocity centers,
       and `sigma` is the basis function width. The return value f is a vector
       with activations.
       
       `s` is a vector containing the state, or may be a matrix in which each
       row specifies a state. In that case, `f` is a matrix where each row
       contains the activation for a row in `s`.
    """
       
    s = np.atleast_2d(s)
    pd = s[:, None, 0] - p.flatten() 
    dist = np.sqrt(pd**2 + ((s[:, None, 1] - v.flatten())/(8/pi))**2)
    return np.squeeze(scipy.stats.norm.pdf(dist, 0, sigma))

def rbfprojector(nbasis, sigma):
    """Returns function that projects states onto Gaussian radial basis function features.
    
       feature = rbfprojector(nbasis, sigma) returns a function
           f = feature(s)
       that projects a state `s` onto a Gaussian RBF feature vector `f`. `nbasis` is the number
       of basis functions per dimension, while `sigma` is their width.
       
       If `s` is a matrix where each row is a state, `f` is a matrix where each row
       contains the feature vector for a row of `s`.
       
       EXAMPLE
           >>> feature = rbfprojector(3, 2)
           >>> print(feature([0, 0, 0]))
           [0.01691614 0.05808858 0.05808858 0.19947114 0.01691614 0.05808858]
    """

    p, v = np.meshgrid(np.linspace(-pi, pi-(2*pi)/(nbasis-1), nbasis-1), np.linspace(-8, 8, nbasis))
    return lambda x: __gaussrbf(x, p, v, sigma)