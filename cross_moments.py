# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:42:04 2018

@author: Yixian
"""

from simulation import (DELTA, T, X_0)
from single_moments import f_func, g_func, h_func
from scipy.stats import norm
import numpy as np
from numpy.linalg import inv
import simulation
from statsmodels.sandbox.distributions.extras import mvnormcdf

def _cross_moments_inner(ksi, eta, means, stds, rho, handle):
    ''''''

    mean, cov = simulation.param_converter(means, stds, rho)
    assert handle.shape == (2, 1)
    adj_mean = np.squeeze(mean + cov.dot(handle), axis=1)
    assert adj_mean.shape == (2,), "{}".format(adj_mean.shape)
    
    mut = 0.5*(((adj_mean.T).dot(inv(cov))).dot(adj_mean)) - 0.5*(((mean.T).dot(inv(cov))).dot(mean))
    upper = np.zeros((2,))
    upper[0] = np.log(ksi/X_0[0])
    upper[1] = np.log(eta/X_0[1])
    ret = - 1 + g_func(ksi/X_0[0], mean=adj_mean[0], sigma2=stds[0]**2) \
            + g_func(eta/X_0[1], mean=adj_mean[1], sigma2=stds[1]**2) \
            + mvnormcdf(upper, adj_mean, cov)
    ret *= np.exp(mut)
    return ret

def H_func(ksi, eta, means, stds, rho):
    ''''''

    handle = np.zeros((2, 1))
    ret = _cross_moments_inner(ksi, eta, means, stds, rho, handle)
    return ret

def G_func(ksi, eta, means, stds, rho):
    ''''''

    handle = np.ones((2, 1))
    ret = _cross_moments_inner(ksi, eta, means, stds, rho, handle)
    return ret

def F_func(ksi, eta, means, stds, rho, side=None):
    ''''''

    assert side in [1,2], "Unrecognized side {}".format(side)
    handle = np.zeros((2, 1))
    if side == 1:
        handle[0] = 1
    else:
        handle[1] = 1
    ret = _cross_moments_inner(ksi, eta, means, stds, rho, handle)
    return ret

def simulation_check_funcs(ksi, eta, num_samples, means, stds, rho):
    ''''''

    X_mnT = simulation.simulate_xt(num_samples, means, stds, rho)
    H = H_func(ksi, eta, means, stds, rho)
    F1 = F_func(ksi, eta, means, stds, rho, side=1)
    F2 = F_func(ksi, eta, means, stds, rho, side=2)
    G = G_func(ksi, eta, means, stds, rho)
    H_simu, F1_simu, F2_simu, G_simu = 0, 0, 0, 0
    for n in range(num_samples):
        if (X_mnT[0][n] >= ksi) and (X_mnT[1][n] >= eta):
            H_simu += 1
            F1_simu += X_mnT[0][n]/X_0[0]
            F2_simu += X_mnT[1][n]/X_0[1]
            G_simu += (X_mnT[0][n]/X_0[0]) * (X_mnT[1][n]/X_0[1])
    H_simu /= num_samples
    G_simu /= num_samples
    F1_simu /= num_samples
    F2_simu /= num_samples
    print("H: {}".format(H))
    print("H_simu: {}".format(H_simu))
    print("Diff: {}".format(abs(H-H_simu)))
    print("G: {}".format(G))
    print("G_simu: {}".format(G_simu))
    print("Diff: {}".format(abs(G-G_simu)))
    print("F1: {}".format(F1))
    print("F1_simu: {}".format(F1_simu))
    print("Diff: {}".format(abs(F1-F1_simu)))
    print("F2: {}".format(F2))
    print("F2_simu: {}".format(F2_simu))
    print("Diff: {}".format(abs(F2-F2_simu)))

if __name__ == "__main__":
    num_samples = 1000000
    means = [0.5, 1]
    stds = [2, 3]
    rho = 0.5
    ksi = 107
    eta = 103
    # TODO: still have some non-trivial errors
    simulation_check_funcs(ksi, eta, num_samples, means, stds, rho)




#    X_mnT = simulation.simulate_xt(num_samples, means, stds, rho)
#    Y_T = np.log(X_mnT/100)
#    mean, cov = simulation.param_converter(means, stds, rho)
#    mean = np.squeeze(mean, axis=1)
#    upper = np.zeros((2,))
#    upper[0] = np.log(ksi/100)
#    upper[1] = np.log(eta/100)
#    simu = 0
#    for n in range(num_samples):
#        if (Y_T[0][n] <= upper[0]) and (Y_T[1][n] <= upper[1]):
#            simu += 1
#    simu /= num_samples
#    ana = mvnormcdf(upper, mean, cov)
    
    
    