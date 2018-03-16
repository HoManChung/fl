# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 20:39:04 2017

@author: Yixian
"""

import numpy as np
import common
import fund
import matplotlib.pyplot as plt

#num_samples = 1000000
#return_m = 0.10
#return_sd = 0.08
#
#X_T = common._simulate_xt(num_samples, return_m, return_sd)
#
#payoffs = fund._fund_payoff(X_T, 0.15, 0.0)
#mean_before = np.mean(payoffs)
#std_before = np.std(payoffs)
#sharpe_star = mean_before/std_before
#
#payoffs = fund._fund_payoff(X_T, 0.25, 0.05)
#mean_after = np.mean(payoffs)
#std_after = np.std(payoffs)

#beta = 0.1
#alphas = np.arange(0, 0.99, 0.01)
#sharpe = np.zeros_like(alphas)
#for idx, alpha in enumerate(alphas):
#    payoffs = fund._fund_payoff(X_T, alpha, beta)
#    mean = np.mean(payoffs)
#    std = np.std(payoffs)
#    sharpe[idx] = mean/std

x = np.arange(0, 2, 0.1)
y = x ** 2
plt.plot(x, x, 'r-', x, y, 'b--' )
mu = 0.16
mu_per = mu*100
title1 = r"$\tilde{\mu}=$"
title2 = r"{}%".format(mu_per)
title = title1 + title2
plt.title(title)
plt.show()
