# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 21:35:57 2017

@author: Yixian
"""

import numpy as np
import matplotlib.pyplot as plt
from common import R
import fund
import pricing_combined

LAMBDA_FIG2 = 5.0
MU_FIG3 = 0.15
SIGMA_FIG3 = 0.32
VOL_THRESHOLD_FIG3 = 0.30

def figure0():
    '''Example of fitting the indifference
    curve for points that manager specifies.'''

    fig, ax = plt.subplots()
    betas = np.arange(0.0, 0.101, 0.001)
    alphas = []
    for beta in betas:
        alphas.append(fund.alpha_as_beta(beta, r=-0.05, sigma2=0.07**2))
    ax.plot(betas, alphas, c='k')
    ax.plot(0.03, 0.26, 'k^')
    ax.plot(0.04, 0.34, 'k^')
    ax.plot(0.05, 0.38, 'k^')
    vals = ax.get_xticks()
    ax.set_xticklabels(['{:2.0f}%'.format(x*100) for x in vals])
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:2.0f}%'.format(y*100) for y in vals])
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\alpha$")
    plt.savefig('figure0.eps', format='eps', dpi=600)
    plt.show()

def figure1(ttype):
    '''From the fund's perspective, 
    curves of alpha as a function of 
    fund's vol.
    '''

    fig, ax = plt.subplots()
    if ttype is "a":
        sigmas = [0.12, 0.20, 0.35]
        betas = np.arange(0.0, 0.31, 0.01)
        styles = ['-', '--', '-.']
        for sigma, s in zip(sigmas, styles):
            alphas = []
            for beta in betas:
                alphas.append(fund.alpha_as_beta(beta, sigma2=sigma**2))
            ax.plot(betas, alphas, c='k', linestyle=s, label=r"$\sigma={:2.0f}$%".format(sigma*100))
    elif ttype is "b":
        rs = [0, -0.05, -0.1]
        betas = np.arange(0.0, 0.051, 0.001)
        styles = ['-', '--', '-.']
        for r, s in zip(rs, styles):
            alphas = []
            for beta in betas:
                alphas.append(fund.alpha_as_beta(beta, r=r, sigma2=0.12**2))
            ax.plot(betas, alphas, c='k', linestyle=s, label=r"$r={:2.0f}$%".format(r*100))
    else:
        raise NotImplementedError()
    
    vals = ax.get_xticks()
    ax.set_xticklabels(['{:2.0f}%'.format(x*100) for x in vals])
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:2.0f}%'.format(y*100) for y in vals])
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\alpha$")
    plt.legend()
    #plt.title(r"$\alpha$ as a function of $\beta$ for different $\sigma_f$")
    plt.savefig('figure1{}.eps'.format(ttype), format='eps', dpi=600)
    plt.show()
        

def figure2(ttype):
    '''Combined pricing, optimal alpha and beta 
    for three different mus and varying sigmas
    '''

    mus = { "a":0.06, "b":0.10, "c":0.16 }
    mu = mus[ttype]
    sigmas = np.arange(0.03, 0.25, 0.001)
    
    llambdas = [LAMBDA_FIG2]
    alphas = []
    betas = []
    for sigma in sigmas:
        res = pricing_combined.pricing_combined(llambdas,
                                                mu=mu,
                                                sigma2=sigma**2)
        alphas.append(res[4][0])
        betas.append(res[3][0])

    fig, ax = plt.subplots()
    ax.plot(sigmas, alphas, c='k', linestyle='-', label=r"$\alpha$")
    ax.plot(sigmas, betas,  c='k', linestyle='--', label=r"$\beta$")

    vals = ax.get_xticks()
    ax.set_xticklabels(['{:2.0f}%'.format(x*100) for x in vals])
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:2.0f}%'.format(y*100) for y in vals])
    plt.xticks(np.arange(0.03, 0.25, 0.07))
    plt.xlabel(r"$\tilde{\sigma}$")
    plt.legend()
    title1 = r"$\tilde{\mu}=$"
    title2 = r"{:2.0f}%".format(mu*100)
    title = title1 + title2
    plt.title(title)
    fig_str = 'figure2{}.eps'.format(ttype)
    plt.savefig(fig_str, format='eps', dpi=600)
    plt.show()

    return alphas, betas

def figure3():
    '''Plot Sharpe ratio and variance, varying llambda.'''

    llambdas = np.arange(0.1, 5.1, 0.1)
    res = pricing_combined.pricing_combined(llambdas,
                                            mu=MU_FIG3,
                                            sigma2=SIGMA_FIG3**2)
    e_stars = res[-2]
    vol_stars = res[-1]
    sharpe = (e_stars-(1+R))/vol_stars
    vol_thres = [VOL_THRESHOLD_FIG3]*len(llambdas)
    
    fig, ax = plt.subplots()
    ax.plot(llambdas, sharpe,    'k^', label="Sharpe ratio", markersize=4)
    ax.plot(llambdas, vol_stars, 'k.', label="vol", markersize=4)
    ax.plot(llambdas, vol_thres, 'k--', label="vol_thres")

    vals = ax.get_yticks()
    ax.set_yticklabels(['{:2.0f}%'.format(y*100) for y in vals])
    plt.xticks(np.arange(0.0, 5.0, 1.0))
    plt.xlabel(r"$\lambda$")
    plt.legend()
    plt.savefig('figure3.eps', format='eps', dpi=600)
    plt.show()

    return sharpe, vol_stars     

if __name__ == "__main__":
    figure0()
    for ttype in ['a', 'b']:
        figure1(ttype)
    for ttype in ["a", "b", "c"]:
        _ = figure2(ttype)
    _ = figure3()
