# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:40:16 2018

@author: Yixian
"""

from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import optimization

def figure0():
    '''Example of fitting the indifference
    curve for points that manager specifies.'''

    fig, ax = plt.subplots()
    betas = np.arange(0.025, 0.055, 0.001)
    alphas1, alphas2 = [], []
    x = [0.02, 0.03, 0.04, 0.05, 0.06]
    y = [0.20, 0.26, 0.30, 0.38, 0.50]
    linear = interpolate.interp1d(x, y, kind='linear')
    quadratic = interpolate.interp1d(x, y, kind='quadratic')
    for beta in betas:
        alphas1.append(linear(beta))
        alphas2.append(quadratic(beta))
    ax.plot(betas, alphas1, c='k')
    ax.plot(betas, alphas2, c='k', linestyle=':')
    ax.plot(x[1], y[1], 'k^')
    ax.plot(x[2], y[2], 'k^')
    ax.plot(x[3], y[3], 'k^')
    vals = ax.get_xticks()
    ax.set_xticklabels(['{:2.1f}%'.format(x*100) for x in vals])
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:2.0f}%'.format(y*100) for y in vals])
    plt.xlabel(r"loss protection")
    plt.ylabel(r"incentive")
    plt.savefig('figure00.eps', format='eps', dpi=600)
    plt.show()

def illustrate_expectation(ret):
    ''''''

    fig, ax = plt.subplots()
    lambdas = ret['lambda'].values
    e0 = ret['expectation0'].values
    e = ret['expectation'].values

    ax.plot(lambdas, e0, c='k', linestyle='--', label=r"traditional scheme")
    ax.plot(lambdas, e, c='k', linestyle='-', label=r"with loss protection")

    vals = ax.get_xticks()
    ax.set_xticklabels(['{:.2f}'.format(x) for x in vals])
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:2.0f}%'.format(y*100) for y in vals])
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"expectation returns in %")
    plt.legend()
    plt.savefig('expectations.eps', format='eps', dpi=600)
    plt.show()
    
def illustrate_std(ret):
    ''''''

    fig, ax = plt.subplots()
    lambdas = ret['lambda'].values
    std0 = np.sqrt(ret['variance0'].values)
    std = np.sqrt(ret['variance'].values)

    ax.plot(lambdas, std0, c='k', linestyle='--', label=r"traditional scheme")
    ax.plot(lambdas, std, c='k', linestyle='-', label=r"with loss protection")

    vals = ax.get_xticks()
    ax.set_xticklabels(['{:.2f}'.format(x) for x in vals])
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:2.1f}'.format(y*100) for y in vals])
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"volatility in %")
    plt.legend()
    #plt.title(r"$\alpha$ as a function of $\beta$ for different $\sigma_f$")
    plt.savefig('volatility.eps', format='eps', dpi=600)
    plt.show()
    
def illustrate_weights(ret):
    ''''''

    fig, ax = plt.subplots()
    lambdas = ret['lambda'].values
    weights0 = ret['weights0'].values.tolist()
    weights = ret['weights'].values.tolist()

    wd1, wd2, wd3 = [], [], []
    for w0, w in zip(weights0, weights):    
        wd1.append(w[0] - w0[0])
        wd2.append(w[1] - w0[1])
        wd3.append(w[2] - w0[2])

    ax.plot(lambdas, wd1, c='k', linestyle='--', label=r"Fund 1")
    ax.plot(lambdas, wd2, c='k', linestyle='-', label=r"Fund 2")
    ax.plot(lambdas, wd3, c='k', linestyle='-.', label=r"Fund 3")

    vals = ax.get_xticks()
    ax.set_xticklabels(['{:.2f}'.format(x) for x in vals])
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:2.0f}'.format(y*100) for y in vals])
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"weight difference in %")
    plt.legend()
    #plt.title(r"$\alpha$ as a function of $\beta$ for different $\sigma_f$")
    plt.savefig('weights.eps', format='eps', dpi=600)
    plt.show()
    
def illustrate_protection(ret):
    ''''''

    fig, ax = plt.subplots()
    lambdas = ret['lambda'].values
    betas = ret['beta'].values.tolist()

    beta1 = [ b[0] for b in betas ]
    beta2 = [ b[1] for b in betas ]
    beta3 = [ b[2] for b in betas ]

    ax.plot(lambdas, beta1, c='k', linestyle='--', label=r"Fund 1")
    ax.plot(lambdas, beta2, c='k', linestyle='-', label=r"Fund 2")
    ax.plot(lambdas, beta3, c='k', linestyle='-.', label=r"Fund 3")

    vals = ax.get_xticks()
    ax.set_xticklabels(['{:.2f}'.format(x) for x in vals])
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:2.0f}'.format(y*100) for y in vals])
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"loss protections in %")
    plt.legend()
    #plt.title(r"$\alpha$ as a function of $\beta$ for different $\sigma_f$")
    plt.savefig('protection.eps', format='eps', dpi=600)
    plt.show()
    


if __name__ == "__main__":
    fund_specs = { 0 : [ (0.15, 0.0), (0.16, 0.01), (0.17, 0.02), (0.19, 0.03), (0.22, 0.05) ], 
                   1 : [ (0.15, 0.0), (0.165, 0.01), (0.17, 0.02), (0.18, 0.05) ], 
                   2 : [ (0.15, 0.0), (0.17, 0.01), (0.19, 0.03), (0.21, 0.04), (0.24, 0.05), (0.27, 0.10) ] }
    means = [0.12, 0.12, 0.20]
    stds = [0.25, 0.27, 0.35]
    rhos = {(0, 1) : 0.1, (0, 2) : 0.1, (1, 2) : 0.1}
    lambdas = np.arange(0, 0.17, 0.01)
#    ret = optimization.solve_mean_variance(lambdas, means, stds, rhos, fund_specs)
    figure0()
    illustrate_expectation(ret)
    illustrate_std(ret)
    illustrate_weights(ret)
    illustrate_protection(ret)
    
    