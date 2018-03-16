# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 22:53:05 2017

@author: Yixian
"""

from scipy.stats import norm
import numpy as np

DELTA = 0.015 #1.5%
T = 1
R = 0.01 #1% #TODO: change it back
X_0 = 100

def _simulate_xt(num_samples, mean=None, std=None):
    '''Simulate samples of X_T'''

    assert mean is not None and std is not None
    np.random.seed(seed=0)
    sample = np.random.normal(0, 1, num_samples)
    sample = sample*std + mean
    X_T = np.exp(sample)*X_0

    return X_T

def _f_func_generator(mean=None, sigma2=None):
    '''E(X*1(X>ksi))'''

    assert mean is not None and sigma2 is not None
    def inner(ksi):
        ret = np.exp(mean+0.5*sigma2)
        temp = (-np.log(ksi)+(mean+sigma2))/np.sqrt(sigma2)
        ret *= (norm.cdf(temp))
        return ret
    return inner

def _g_func_generator(mean=None, sigma2=None):
    '''E(1(X>ksi))'''

    assert mean is not None and sigma2 is not None
    def inner(ksi):
        temp = (np.log(ksi)-mean) / np.sqrt(sigma2)
        return 1 - norm.cdf(temp)
    return inner

def _h_func_generator(mean=None, sigma2=None):
    '''E(X^2*1(X>ksi))'''

    assert mean is not None and sigma2 is not None
    def inner(ksi):
        ret = np.exp(2*mean+2*sigma2)
        temp = (-np.log(ksi)+(mean+2*sigma2))/np.sqrt(sigma2)
        ret *= (norm.cdf(temp))
        return ret
    return inner
