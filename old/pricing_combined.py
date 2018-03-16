# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:33:07 2017

@author: Yixian
"""

import numpy as np
import matplotlib.pyplot as plt
from fund import BETA_UPPER_BOUND
from investor import (MU, SIGMA2)

import fund
import investor

GRID_SIZE = 0.001 # 0.1% in search for beta
BETAS = np.arange(GRID_SIZE, BETA_UPPER_BOUND+GRID_SIZE, GRID_SIZE)

def pricing_combined(llambdas, mu=MU, sigma2=SIGMA2):
    '''Provide optimal values of alpha
    and beta, combining fund and investor's
    perspective.'''

    expectations = np.array([])
    variances = np.array([])
    vols = np.array([])
    b_stars = np.array([])
    a_stars = np.array([])
    e_stars = np.array([])
    vol_stars = np.array([])
    for b in BETAS:
        a = fund.alpha_as_beta(b)
        e, v = investor.objective(a, b, mu=mu, sigma2=sigma2)
        expectations = np.append(expectations, e)
        variances = np.append(variances, v)
        vols = np.append(vols, np.sqrt(v))

    for llambda in llambdas:
        objs = expectations - llambda*variances
        b_star = BETAS[np.argmax(objs)]
        b_stars = np.append(b_stars, b_star)
        a_star = fund.alpha_as_beta(b_star)
        a_stars = np.append(a_stars, a_star)
        e_star = expectations[np.argmax(objs)]
        e_stars = np.append(e_stars, e_star)
        vol_star = vols[np.argmax(objs)]
        vol_stars = np.append(vol_stars, vol_star)

    res = (
            expectations,
            variances,
            vols,
            b_stars,
            a_stars,
            e_stars,
            vol_stars
            )

    return res

def visualization_pricing_results(betas, alphas, objs):
    '''Plot alphas and investor obj
    against betas.'''

    plt.plot(betas, alphas, 'r--')
    plt.show()

if __name__ == "__main__":
    llambdas = [ 5.0 ]
    mu = 0.10
    sigma = 0.25
    res = pricing_combined(llambdas, mu=mu, sigma2=sigma**2)
    expectations, vols = res[0], res[2]
    plt.plot(BETAS, expectations, 'r', BETAS, vols, 'b')
    plt.show()