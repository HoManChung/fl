# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 22:21:31 2018

@author: Yixian
"""

DELTA = 0.015 #1.5%
X_m0 = 100
X_n0 = 100

from scipy.stats import norm
import numpy as np

def _param_converter(means, stds, rho):
    ''''''

    assert len(means) == 2
    mean = np.array(means)
    assert len(stds) == 2
    assert rho >= -1 and rho <= 1
    var = np.array([[stds[0]**2, stds[0]*stds[1]*rho], [stds[0]*stds[1]*rho, stds[1]**2]])
    return mean, var    

def _simulate_xt(num_samples, means, stds, rho):
    '''Simulate samples of X_T=(X_mT, X_mT)'''

    mean, _ = _param_converter(means, stds, rho)
    std = np.array([[stds[0]*rho, stds[0]*np.sqrt(1-rho*rho)], [stds[1], 0]])
    np.random.seed(seed=0)
    sample = np.random.normal(0, 1, (2, num_samples))
    sample = std.dot(sample) + mean[:, np.newaxis]
    X_T = np.array([[X_m0, 0], [0, X_n0]]).dot(np.exp(sample))

    return X_T

def H_func(ksi, eta, means, stds, rho):
    ''''''

    pass
    

if __name__ == "__main__":
    num_samples = 1000000
    means = [0.5, 1]
    stds = [2, 3]
    rho = 0.5
    X_T = _simulate_xt(num_samples, means, stds, rho)
    Y_T = np.log(X_T/100)
    print("mean:{}".format(np.mean(Y_T, axis=1)))
    print("std:{}".format(np.std(Y_T, axis=1)))
    print("corr:{}".format(np.corrcoef(Y_T[0], y=Y_T[1])[0][1]))