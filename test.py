# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 09:59:35 2018

@author: Yixian
"""

import numpy as np
import simulation

means = [0.5, 1]
stds = [2, 3]
rho = 0.5

mean, cov = simulation.param_converter(means, stds, rho)
std1 = np.array([[stds[0]*rho, stds[0]*np.sqrt(1-rho*rho)], [stds[1], 0]])
std2 = np.linalg.cholesky(cov)
assert (std1 == std2).all()