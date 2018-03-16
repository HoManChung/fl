# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:50:02 2017

@author: Yixian
"""

import numpy as np
import common

from common import (DELTA, T, X_0)

MU = 0.1 #10%
SIGMA2 = 0.4*0.4 # 40%

_f_func = common._f_func_generator(mean=(MU-0.5*SIGMA2)*T, sigma2=SIGMA2*T)
_g_func = common._g_func_generator(mean=(MU-0.5*SIGMA2)*T, sigma2=SIGMA2*T)
_h_func = common._h_func_generator(mean=(MU-0.5*SIGMA2)*T, sigma2=SIGMA2*T)

def eterm1(alpha, mu=MU, sigma2=SIGMA2):
    '''Term 1 for expectation'''

    deltat = DELTA * T
    alphat = alpha * T

    _f_func = common._f_func_generator(mean=(mu-0.5*sigma2)*T, sigma2=sigma2*T)
    _g_func = common._g_func_generator(mean=(mu-0.5*sigma2)*T, sigma2=sigma2*T)

    term1a = (1-alphat)*_f_func(1+deltat)
    term1b = deltat*(1-alphat)*_g_func(1+deltat)
    return term1a - term1b

def eterm2(beta, mu=MU, sigma2=SIGMA2):
    '''Term 2 for expectation'''

    betat = beta * T
    deltat = DELTA * T

    _g_func = common._g_func_generator(mean=(mu-0.5*sigma2)*T, sigma2=sigma2*T)

    term2a = _g_func(1-betat+deltat)
    term2b = _g_func(1+deltat)
    return term2a - term2b

def eterm3(beta, mu=MU, sigma2=SIGMA2):
    '''Term 3 for expectation'''

    betat = beta * T
    deltat = DELTA * T

    _f_func = common._f_func_generator(mean=(mu-0.5*sigma2)*T, sigma2=sigma2*T)
    _g_func = common._g_func_generator(mean=(mu-0.5*sigma2)*T, sigma2=sigma2*T)

    ksi = 1-betat+deltat
    term3a = np.exp(mu*T) - _f_func(ksi)
    term3b = (betat-deltat)*(1-_g_func(ksi))
    return term3a + term3b

def sterm1(alpha, mu=MU, sigma2=SIGMA2):
    '''Term 1 for second-order moment'''

    deltat = DELTA * T
    alphat = alpha * T

    _f_func = common._f_func_generator(mean=(mu-0.5*sigma2)*T, sigma2=sigma2*T)
    _g_func = common._g_func_generator(mean=(mu-0.5*sigma2)*T, sigma2=sigma2*T)

    term1a = (1-alphat)*(1-alphat)*_h_func(1+deltat)
    term1b = 2*deltat*(1-alphat)*(1-alphat)*_f_func(1+deltat)
    term1c = deltat*deltat*(1-alphat)*(1-alphat)*_g_func(1+deltat)

    return term1a - term1b + term1c

def sterm2(beta, mu=MU, sigma2=SIGMA2):
    '''Term 2 for second-order moment'''

    betat = beta * T
    deltat = DELTA * T

    _g_func = common._g_func_generator(mean=(mu-0.5*sigma2)*T, sigma2=sigma2*T)

    term2a = _g_func(1-betat+deltat)
    term2b = _g_func(1+deltat)
    return term2a - term2b

def sterm3(beta, mu=MU, sigma2=SIGMA2):
    '''Term 3 for second-order moment'''

    betat = beta * T
    deltat = DELTA * T

    _f_func = common._f_func_generator(mean=(mu-0.5*sigma2)*T, sigma2=sigma2*T)
    _g_func = common._g_func_generator(mean=(mu-0.5*sigma2)*T, sigma2=sigma2*T)
    _h_func = common._h_func_generator(mean=(mu-0.5*sigma2)*T, sigma2=sigma2*T)

    ksi = 1-betat+deltat
    term3a = np.exp(2*mu*T+SIGMA2*T) - _h_func(ksi)
    term3b = 2*(betat-deltat)*(np.exp(mu*T)-_f_func(ksi))
    term3c = (betat-deltat) * (betat-deltat) * (1-_g_func(ksi))
    return term3a + term3b + term3c

def expectation(alpha, beta, mu=MU, sigma2=SIGMA2):
    '''The expectation of payoff from investor's perspective.'''

    component1 = eterm1(alpha, mu=mu, sigma2=sigma2)
    component2 = eterm2(beta, mu=mu, sigma2=sigma2)
    component3 = eterm3(beta, mu=mu, sigma2=sigma2)

    mul = X_0
    ret = component1
    ret += component2
    ret += component3
    ret *= mul

    return ret

def variance(alpha, beta, mu=MU, sigma2=SIGMA2):
    '''The variance of payoff from investor's perspective.'''

    component1 = sterm1(alpha, mu=mu, sigma2=sigma2)
    component2 = sterm2(beta, mu=mu, sigma2=sigma2)
    component3 = sterm3(beta, mu=mu, sigma2=sigma2)

    mul = X_0 * X_0
    ret = component1
    ret += component2
    ret += component3
    ret *= mul
    e = expectation(alpha, beta, mu=mu, sigma2=sigma2)
    subtract = e * e
    ret -= subtract

    return ret

def objective(alpha, beta, mu=MU, sigma2=SIGMA2):
    '''Calc mean and var of return for investor'''

    e = expectation(alpha, beta, mu=mu, sigma2=sigma2)
    e = e/X_0
    v = variance(alpha, beta, mu=mu, sigma2=sigma2)
    v = v/(X_0*X_0)
    return e, v

def _investor_payoff(x, alpha, beta):
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

def run_simulation_check(num_samples, mu=MU, sigma2=SIGMA2):
    '''Check expectation and variance
    function by simulation'''

    alphas = np.arange(0, 0.31, 0.01)
    betas = np.arange(0, 0.31, 0.01)
    M, N = len(alphas), len(betas)
    expectations = np.zeros((M, N))
    simulated_e = np.zeros((M, N))
    variances = np.zeros((M, N))
    simulated_v = np.zeros((M, N))
    X_T = common._simulate_xt(num_samples,
                              mean=(mu-0.5*sigma2)*T,
                              std=np.sqrt(sigma2*T))

    for m in range(M):
        for n in range(N):
            alpha = alphas[m]
            beta = betas[n]
            print("alpha: {}, beta: {}".format(alpha, beta))
            expectations[m][n] = expectation(alpha, beta, mu=mu, sigma2=sigma2)
            simulated_e[m][n] = np.mean(_investor_payoff(X_T, alpha, beta))
            variances[m][n] = variance(alpha, beta, mu=mu, sigma2=sigma2)
            simulated_v[m][n] = np.var(_investor_payoff(X_T, alpha, beta))

    diff_e = np.abs(expectations-simulated_e)
    diff_v = np.abs(variances-simulated_v)
    return diff_e, diff_v, expectations, simulated_e, variances, simulated_v

if __name__ == "__main__":
    num_samples = 1000000
    ret = run_simulation_check(num_samples)
    diff_e, diff_v, expectations, simulated_e, variances, simulated_v = ret
