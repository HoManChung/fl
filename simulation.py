# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 22:21:31 2018

@author: Yixian
"""

DELTA = 0.015 #1.5%
T = 1
X_0 = [100, 100] #=[X_m0, X_n0]

import numpy as np

def param_converter(means, stds, rhos, ndim=2):
    '''Only works for 2dim'''

    if ndim == 2:
        raise NotImplementedError()
    elif ndim == 3:    
        assert len(means) == 3
        assert len(stds) == 3
        assert len(rhos) == 3
        
        mean = np.array(means)
        mean = mean[:, np.newaxis]
        
        cov = np.array([[stds[0]*stds[0], rhos[(0, 1)]*stds[0]*stds[1], rhos[(0, 2)]*stds[0]*stds[2]],
                        [rhos[(0, 1)]*stds[1]*stds[0], stds[1]*stds[1], rhos[(1, 2)]*stds[1]*stds[2]],
                        [rhos[(0, 2)]*stds[2]*stds[0], rhos[(1, 2)]*stds[1]*stds[2], stds[2]*stds[2]]])

    return mean, cov

def simulate_xt(num_samples, means=None, stds=None, rho=None):
    '''Simulate samples of X_mnT=(X_mT, X_mT)'''

    assert means is not None and stds is not None and rho is not None
    mean, _ = param_converter(means, stds, rho)
    std = np.array([[stds[0]*rho, stds[0]*np.sqrt(1-rho*rho)], [stds[1], 0]])
    np.random.seed(seed=0)
    sample = np.random.normal(0, 1, (2, num_samples))
    sample = std.dot(sample) + mean
    X_mnT = np.array([[X_0[0], 0], [0, X_0[1]]]).dot(np.exp(sample))

    return X_mnT

def simulate_xt_ndim(num_samples, mean, cov, init_values):
    ''''''

    assert type(mean) == np.ndarray and type(cov) == np.ndarray, "Wrong types!"
    assert (mean.shape[0] == cov.shape[0]) and (cov.shape[0] == cov.shape[1]) and (len(init_values)), "Dimension mismatch!"
    assert len(mean.shape) == 2 and mean.shape[1]==1, "Need two-dim mean"
    ndim = mean.shape[0]
    std = np.linalg.cholesky(cov)

    np.random.seed(seed=0)
    sample = np.random.normal(0, 1, (ndim, num_samples))
    sample = std.dot(sample) + mean
    X_T = np.exp(sample)
    X_T = np.diag(init_values).dot(X_T)
    
    return X_T
    
    
    

#TODO: get rid of one _investor_payoff_inner
def _investor_payoff_inner0(x, X_0, alpha, beta):
    '''payoff function from the perspective of investor'''

    alphat = alpha * T
    betat = beta * T
    deltat = DELTA * T
    payoff = np.zeros(x.shape)
    N = len(x)
    for n in range(N):
        X_T = x[n]
        if X_T >= (1+deltat)*X_0:
            payoff[n] = X_T - (alphat*(X_T-deltat*X_0) + deltat*X_0)
        elif X_T >= (1-betat+deltat)*X_0:
            payoff[n] = X_0
        else:
            payoff[n] = X_T+betat*X_0-deltat*X_0

    return payoff

def _investor_payoff_inner1(x, X_0, alpha, beta):
    '''payoff function from the perspective of investor'''

    alphat = alpha * T
    betat = beta * T
    deltat = DELTA * T
    payoff = np.zeros(x.shape)
    N = len(x)
    for n in range(N):
        X_T = x[n]
        if X_T >= (1+deltat)*X_0:
            payoff[n] = X_T - (alphat*(X_T-deltat*X_0) + deltat*X_0)
        elif X_T >= (1-betat+deltat)*X_0:
            payoff[n] = X_0
        else:
            payoff[n] = X_T+betat*X_0-deltat*X_0

    return payoff


def investor_payoffs(X_T, alphas, betas, X_0=X_0):
    ''''''

    assert X_T.shape[0] == len(alphas)
    assert X_T.shape[0] == len(betas)

    payoffs = np.zeros(X_T.shape)
    for r in range(payoffs.shape[0]):
        payoffs[r, :] = _investor_payoff_inner0(X_T[r], X_0[r], alphas[r], betas[r])
#    payoffs[0, :] = _investor_payoff_inner0(X_mnT[0], X_0[0], alphas[0], betas[0])
#    payoffs[1, :] = _investor_payoff_inner1(X_mnT[1], X_0[1], alphas[1], betas[1])
            
    return payoffs
    

if __name__ == "__main__":
    num_samples = 1000000
#    means = [0.5, 1]
#    stds = [2, 3]
#    rho = 0.5
#    X_mnT = simulate_xt(num_samples, means, stds, rho)

    means = [0.05, 0.07, 0.20]
    stds = [0.10, 0.15, 0.35]
    rhos = {(0, 1) : 0.3, (0, 2) : -0.5, (1, 2) : 0.2} 
    mean, cov = param_converter(means, stds, rhos, ndim=3)
    X_mnT = simulate_xt_ndim(num_samples, mean, cov, [1]*3)
    Y_T = np.log(X_mnT)
    print("mean:{}".format(np.mean(Y_T, axis=1)))
    print("std:{}".format(np.std(Y_T, axis=1)))
    print("corr01:{}".format(np.corrcoef(Y_T[0], y=Y_T[1])[0][1]))
    print("corr02:{}".format(np.corrcoef(Y_T[0], y=Y_T[2])[0][1]))
    print("corr12:{}".format(np.corrcoef(Y_T[1], y=Y_T[2])[0][1]))
#    payoffs = np.zeros(X_mnT.shape)
#    x = _investor_payoff_inner(X_mnT[0], X_0[0], 0.25, 0.05)