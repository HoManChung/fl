# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 18:46:59 2017

@author: Yixian
"""

import numpy as np
import common

from common import (DELTA, T, R, X_0)

ALPHA0 = 0.15 #15%
BETA0 = 0 #0%
SIGMA2 = 1.5*1.5 # 35%   50*50
BETA_UPPER_BOUND = 0.30 # 50%
ALPHA_UPPER_BOUND = 0.50 # 80%

def term1(alpha, r=R, sigma2=SIGMA2):
    '''Term 1'''

    deltat = DELTA * T
    alphat = alpha * T

    _f_func = common._f_func_generator(mean=(r-0.5*sigma2)*T, sigma2=sigma2*T)
    _g_func = common._g_func_generator(mean=(r-0.5*sigma2)*T, sigma2=sigma2*T)

    term1a = alphat*(_f_func(1+deltat)-deltat*_g_func(1+deltat))
    term1b = deltat*_g_func(1+deltat)
    return term1a + term1b

def term2(beta, r=R, sigma2=SIGMA2):
    '''Term 2'''

    betat = beta * T
    deltat = DELTA * T

    _f_func = common._f_func_generator(mean=(r-0.5*sigma2)*T, sigma2=sigma2*T)
    _g_func = common._g_func_generator(mean=(r-0.5*sigma2)*T, sigma2=sigma2*T)

    term2a = _f_func(1-betat+deltat)-_g_func(1-betat+deltat)
    term2b = _f_func(1+deltat)-_g_func(1+deltat)
    return term2a - term2b

def term3(beta, r=R, sigma2=SIGMA2):
    '''Term 3'''

    betat = beta * T
    deltat = DELTA * T

    _g_func = common._g_func_generator(mean=(r-0.5*sigma2)*T, sigma2=sigma2*T)

    ksi = 1-betat+deltat
    return (deltat-betat)*(1-_g_func(ksi))

def option_px(alpha, beta, r=R, sigma2=SIGMA2):
    '''The price of the first loss option to the fund'''

    component1 = term1(alpha, r=r, sigma2=sigma2)
    component2 = term2(beta, r=r, sigma2=sigma2)
    component3 = term3(beta, r=r, sigma2=sigma2)

    mul = np.exp(-r*T) * X_0
    ret = component1
    ret += component2
    ret += component3
    ret *= mul

    return ret

def _fund_payoff(x, alpha, beta, r=R):
    '''payoff function from the perspective of fund'''

    alphat = alpha * T
    betat = beta * T
    deltat = DELTA * T
    payoff = np.zeros(x.shape)
    N = len(x)
    for n in range(N):
        X_T = x[n]
        if X_T >= (1+deltat)*X_0:
            payoff[n] = alphat*(X_T-deltat*X_0) + deltat*X_0
        elif X_T >= (1-betat+deltat)*X_0:
            payoff[n] = X_T-X_0
        else:
            payoff[n] = -betat*X_0+deltat*X_0

    return payoff*np.exp(-r*T)

def run_simulation_check(num_samples, r=R, sigma2=SIGMA2):
    '''Check option_px function by simulation'''

    alphas = np.arange(0, 0.31, 0.01)
    betas = np.arange(0, 0.26, 0.01)
    M, N = len(alphas), len(betas)
    option_pxs = np.zeros((M, N))
    simulated = np.zeros((M, N))
    X_T = common._simulate_xt(num_samples,
                              mean=(r-0.5*sigma2)*T,
                              std=np.sqrt(sigma2*T))

    for m in range(M):
        for n in range(N):
            alpha = alphas[m]
            beta = betas[n]
            print("alpha: {}, beta: {}".format(alpha, beta))
            option_pxs[m][n] = option_px(alpha, beta, r=R, sigma2=sigma2)
            simulated[m][n] = np.mean(_fund_payoff(X_T, alpha, beta))

    diff = np.abs(option_pxs-simulated)
    return diff, option_pxs, simulated

def alpha_as_beta(beta, r=R, sigma2=SIGMA2, check_tol=1e-8):
    '''alpha as beta so that hedge fund manager is indifferent'''

    assert beta * T <= BETA_UPPER_BOUND
    assert BETA0 == 0, "Formula only applies to the case where BETA0 is 0!"

    _f_func = common._f_func_generator(mean=(r-0.5*sigma2)*T, sigma2=sigma2*T)
    _g_func = common._g_func_generator(mean=(r-0.5*sigma2)*T, sigma2=sigma2*T)

    deltat = DELTA * T

    numerator = term2(BETA0, r=r, sigma2=sigma2) - \
                term2(beta, r=r, sigma2=sigma2) + \
                term3(BETA0, r=r, sigma2=sigma2) - \
                term3(beta, r=r, sigma2=sigma2)
    denominator = T * (_f_func(1+deltat)-deltat*_g_func(1+deltat))
    alpha = ALPHA0 + numerator/denominator

    assert alpha * T <= ALPHA_UPPER_BOUND
    assert abs(option_px(alpha, beta, r=r, sigma2=sigma2)- \
               option_px(ALPHA0, BETA0, r=r, sigma2=sigma2)) < check_tol

    return alpha

if __name__ == "__main__":
    num_samples = 1000000
    diff, option_pxs, simulated = \
        run_simulation_check(num_samples, r=-0.01, sigma2=0.25*0.25)
    betas = np.arange(0, 0.3, 0.01)
    alphas = []
    for beta in betas:
        alphas.append(alpha_as_beta(beta))
