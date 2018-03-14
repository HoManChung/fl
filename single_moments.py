# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 08:50:33 2018

@author: Yixian
"""

from simulation import (DELTA, T, X_0)

from scipy.stats import norm
import numpy as np
import simulation
import statsmodels.sandbox as sb

def f_func(ksi, mean=None, sigma2=None):
    '''E(X*1(X>ksi))'''

    assert mean is not None and sigma2 is not None
    ret = np.exp(mean+0.5*sigma2)
    temp = (-np.log(ksi)+(mean+sigma2))/np.sqrt(sigma2)
    ret *= (norm.cdf(temp))
    return ret

def g_func(ksi, mean=None, sigma2=None):
    '''E(1(X>ksi))'''

    assert mean is not None and sigma2 is not None
    temp = (np.log(ksi)-mean) / np.sqrt(sigma2)
    return 1 - norm.cdf(temp)

def h_func(ksi, mean=None, sigma2=None):
    '''E(X^2*1(X>ksi))'''

    assert mean is not None and sigma2 is not None
    ret = np.exp(2*mean+2*sigma2)
    temp = (-np.log(ksi)+(mean+2*sigma2))/np.sqrt(sigma2)
    ret *= (norm.cdf(temp))
    return ret

def eterm1(alpha, mu, sigma2):
    '''Term 1 for expectation'''

    deltat = DELTA * T
    alphat = alpha * T

    term1a = (1-alphat)*f_func(1+deltat, mean=mu*T, sigma2=sigma2*T)
    term1b = deltat*(1-alphat)*g_func(1+deltat, mean=mu*T, sigma2=sigma2*T)
    return term1a - term1b

def eterm2(beta, mu, sigma2):
    '''Term 2 for expectation'''

    betat = beta * T
    deltat = DELTA * T

    term2a = g_func(1-betat+deltat, mean=mu*T, sigma2=sigma2*T)
    term2b = g_func(1+deltat, mean=mu*T, sigma2=sigma2*T)
    return term2a - term2b

def eterm3(beta, mu, sigma2):
    '''Term 3 for expectation'''

    betat = beta * T
    deltat = DELTA * T

    # TODO: the proof of the current version of the online supplement needs to be re-examined
    ksi = 1-betat+deltat
    term3a = np.exp(mu*T+0.5*sigma2*T) - f_func(ksi, mean=mu*T, sigma2=sigma2*T)
    term3b = (betat-deltat)*(1-g_func(ksi, mean=mu*T, sigma2=sigma2*T))
    return term3a + term3b

def sterm1(alpha, mu, sigma2):
    '''Term 1 for second-order moment'''

    deltat = DELTA * T
    alphat = alpha * T

    term1a = (1-alphat)*(1-alphat)*h_func(1+deltat, mean=mu*T, sigma2=sigma2*T)
    term1b = 2*deltat*(1-alphat)*(1-alphat)*f_func(1+deltat, mean=mu*T, sigma2=sigma2*T)
    term1c = deltat*deltat*(1-alphat)*(1-alphat)*g_func(1+deltat, mean=mu*T, sigma2=sigma2*T)

    return term1a - term1b + term1c

def sterm2(beta, mu, sigma2):
    '''Term 2 for second-order moment'''

    betat = beta * T
    deltat = DELTA * T

    term2a = g_func(1-betat+deltat, mean=mu*T, sigma2=sigma2*T)
    term2b = g_func(1+deltat, mean=mu*T, sigma2=sigma2*T)
    return term2a - term2b

def sterm3(beta, mu, sigma2):
    '''Term 3 for second-order moment'''

    betat = beta * T
    deltat = DELTA * T

    ksi = 1-betat+deltat
    term3a = np.exp(2*mu*T+2*sigma2*T) - h_func(ksi, mean=mu*T, sigma2=sigma2*T)
    term3b = 2*(betat-deltat)*(np.exp(mu*T+0.5*sigma2*T)-f_func(ksi, mean=mu*T, sigma2=sigma2*T))
    term3c = (betat-deltat) * (betat-deltat) * (1-g_func(ksi, mean=mu*T, sigma2=sigma2*T))
    return term3a + term3b + term3c

def expectation(init_value, alpha, beta, mu=None, sigma2=None):
    '''The expectation of payoff from investor's perspective.'''

    assert mu is not None and sigma2 is not None
    component1 = eterm1(alpha, mu=mu, sigma2=sigma2)
    component2 = eterm2(beta, mu=mu, sigma2=sigma2)
    component3 = eterm3(beta, mu=mu, sigma2=sigma2)

    ret = 0
    ret += component1
    ret += component2
    ret += component3
    ret *= init_value

    return ret

def variance(init_value, alpha, beta, mu=None, sigma2=None):
    '''The variance of payoff from investor's perspective.'''

    assert mu is not None and sigma2 is not None
    component1 = sterm1(alpha, mu=mu, sigma2=sigma2)
    component2 = sterm2(beta, mu=mu, sigma2=sigma2)
    component3 = sterm3(beta, mu=mu, sigma2=sigma2)

    ret = 0
    ret += component1
    ret += component2
    ret += component3
    ret *= init_value * init_value
    e = expectation(init_value, alpha, beta, mu=mu, sigma2=sigma2)
    subtract = e * e
    ret -= subtract

    return ret

def run_simulation_check(num_samples, mus, sigma2s, rho):
    '''Check expectation and variance
    function by simulation'''

    alphas = np.arange(0, 0.41, 0.01)#np.arange(0, 0.02, 0.01)#
    betas = np.arange(0, 0.06, 0.01)#np.arange(0, 0.02, 0.01)#
    M, N = len(alphas), len(betas)
    expectations = np.zeros((M, N))
    simulated_e = np.zeros((M, N))
    variances = np.zeros((M, N))
    simulated_v = np.zeros((M, N))
    X_mnT = simulation.simulate_xt(num_samples,
                              means=[mus[0]*T, mus[1]*T],
                              stds=[np.sqrt(sigma2s[0]*T), np.sqrt(sigma2s[1]*T)],
                              rho=0.5)

    for m in range(M):
        for n in range(N):
            alpha = alphas[m]
            beta = betas[n]
            print("alpha: {}, beta: {}".format(alpha, beta))
            payoffs = simulation.investor_payoffs(X_mnT, [alpha, alpha], [beta, beta])
            expectations[m][n] = expectation(X_0[0], alpha, beta, mu=mus[0]*T, sigma2=sigma2s[0]*T)
            simulated_e[m][n] = np.mean(payoffs[0])
            variances[m][n] = variance(X_0[0], alpha, beta, mu=mus[0]*T, sigma2=sigma2s[0]*T)
            simulated_v[m][n] = np.var(payoffs[0])

    diff_e = np.abs(expectations-simulated_e)
    diff_v = np.abs(variances-simulated_v)
    return diff_e, expectations, simulated_e, diff_v, variances, simulated_v

if __name__ == "__main__":
    num_samples = 1000000
    ret = run_simulation_check(num_samples, mus=[0.5, 1], sigma2s=[1, 2.25], rho=0.5)
    diff_e, expectations, simulated_e, diff_v, variances, simulated_v = ret