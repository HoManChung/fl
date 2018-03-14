# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:58:15 2018

@author: Yixian
"""

import numpy as np
import pandas as pd
import simulation

def alpha_beta_combinations(fund_specs):
    '''fund_spec is a dictionary
        - keys: range(num_funds)
        - values: a list of 2dim tuples (alpha, beta)
        
        output of this function is a list of dictionaries
        { 'alpha': [], 'beta': [] }, where the two vectors are of length = num_funds
    '''

    num_funds = len(fund_specs)
    assert num_funds >= 1
    ret = []
    queue = [ { 'alpha': [tup[0]], 'beta':[tup[1]], 'lvl': 0 } for tup in fund_specs[0] ]

    while len(queue)>0:
        this_dict = queue[0]
        queue = queue[1:]
        if this_dict['lvl'] == num_funds-1:
            assert (len(this_dict['alpha']) == num_funds) \
                    and (len(this_dict['beta']) == num_funds)
            ret.append({'alpha': this_dict['alpha'], 'beta':this_dict['beta']})
        else:
            to_add = [ { 'alpha': this_dict['alpha'] + [tup[0]], 
                         'beta': this_dict['beta'] + [tup[1]],   
                         'lvl': this_dict['lvl']+1 } for tup in fund_specs[this_dict['lvl']+1] ]
            queue = queue + to_add

    num_combs = 1
    for v in fund_specs.values():
        num_combs *= len(v)
    assert num_combs == len(ret), "num_combs: {}, ret: {}".format(num_combs, len(ret))
    
    return ret

def produce_mean_cov(means, stds, rhos, fund_specs, num_samples=1000000):
    '''Produce mu and sigma for fund returns under first-loss, 
    using simulation'''
    
    assert len(means) == len(fund_specs)
    mean, cov = simulation.param_converter(means, stds, rhos, ndim=3)
    X_T = simulation.simulate_xt_ndim(num_samples, mean, cov, [1]*3) 
    ab_combs = alpha_beta_combinations(fund_specs)
    ret = {}
    for ab in ab_combs:
        # when X_0 is 1, calc returns
        returns = simulation.investor_payoffs(X_T, ab['alpha'], ab['beta'], X_0=[1]*3)
        mu = np.mean(returns, axis=1)
        sigma = np.cov(returns, rowvar=True)
        assert mu.shape == (3,), "{}".format(mu.shape)
        mu = mu[:,np.newaxis]
        assert sigma.shape == (3, 3), "{}".format(sigma.shape)
        ret[(tuple(ab['alpha']), tuple(ab['beta']))] = { 'mu': mu, 'sigma': sigma}

    return ret

def solve_mean_variance(lambdas, means, stds, rhos, fund_specs, tol=1e-8):
    ''''''

    mu_sigmas = produce_mean_cov(means, stds, rhos, fund_specs)
    num_funds = len(means)
    ones = np.ones((num_funds, 1))
    ret = []
    for l in lambdas:
        this_dict = { 'lambda' : l }
        best_obj = np.inf
        for k, v in mu_sigmas.items():
            mu = v['mu']
            sigma = v['sigma']
            sigma_inv = np.linalg.inv(sigma)        
            term1 = l*sigma_inv.dot(mu)
            nominator = 1 - l*((mu.T).dot(sigma_inv)).dot(ones)
            demon = ((ones.T).dot(sigma_inv)).dot(ones)
            term2 = (nominator/demon)*sigma_inv.dot(ones)
            w = term1 + term2
#            assert (term2.T).dot(ones) == nominator, "{}, {}".format((ones.T).dot(term2), nominator)
#            assert (term1.T).dot(ones) + nominator == 1,  
            assert (np.squeeze((w.T).dot(ones)) - 1)<tol, "{}".format(np.squeeze((w.T).dot(ones)))
            expectation = np.squeeze((mu.T).dot(w))
            variance = np.squeeze(((w.T).dot(sigma)).dot(w))
            obj = variance - 2*l*expectation
            if obj < best_obj:
                best_obj = obj
                this_dict['alpha'] = k[0]
                this_dict['beta'] = k[1]
                this_dict['weights'] = np.squeeze(w)
                this_dict['expectation'] = expectation - 1
                this_dict['variance'] = variance
                this_dict['Sharpe_ratio'] = this_dict['expectation'] / np.sqrt(this_dict['variance'])
        mu0 = mu_sigmas[((0.15, 0.15, 0.15), (0.0, 0.0, 0.0))]['mu']
        sigma0 = mu_sigmas[((0.15, 0.15, 0.15), (0.0, 0.0, 0.0))]['sigma']
        sigma0_inv = np.linalg.inv(sigma0)        
        term1 = l*sigma0_inv.dot(mu0)
        nominator0 = 1 - l*((mu0.T).dot(sigma0_inv)).dot(ones)
        demon0 = ((ones.T).dot(sigma0_inv)).dot(ones)
        term2 = (nominator0/demon0)*sigma0_inv.dot(ones)
        w0 = term1 + term2
        this_dict['weights0'] = np.squeeze(w0)
        this_dict['expectation0'] = np.squeeze((mu0.T).dot(w0)) - 1
        this_dict['variance0'] = np.squeeze(((w0.T).dot(sigma0)).dot(w0))
        this_dict['Sharpe_ratio0'] = this_dict['expectation0'] / np.sqrt(this_dict['variance0'])
        ret.append(this_dict)
    ret = pd.DataFrame.from_dict(ret)

    return ret 
            

if __name__ == "__main__":
    fund_specs = { 0 : [ (0.15, 0.0), (0.17, 0.01), (0.20, 0.02), (0.23, 0.03), (0.30, 0.05) ], 
                   1 : [ (0.15, 0.0), (0.18, 0.01), (0.22, 0.02), (0.27, 0.05) ], 
                   2 : [ (0.15, 0.0), (0.18, 0.01), (0.21, 0.03), (0.27, 0.04), (0.32, 0.05), (0.40, 0.10) ] }
    means = [0.15, 0.15, 0.20]
    stds = [0.25, 0.27, 0.35]
    rhos = {(0, 1) : 0.1, (0, 2) : 0.1, (1, 2) : 0.1}
    lambdas = np.arange(0, 0.21, 0.01)
#    ret = alpha_beta_combinations(fund_specs)
#    ret = produce_mean_cov(means, stds, rhos, fund_specs)
    ret = solve_mean_variance(lambdas, means, stds, rhos, fund_specs)
    