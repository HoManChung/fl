# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:40:16 2018

@author: Yixian
"""

import numpy as np
import optimization



if __name__ == "__main__":
    fund_specs = { 0 : [ (0.15, 0.0), (0.17, 0.01), (0.20, 0.02), (0.23, 0.03), (0.30, 0.05) ], 
                   1 : [ (0.15, 0.0), (0.18, 0.01), (0.22, 0.02), (0.27, 0.05) ], 
                   2 : [ (0.15, 0.0), (0.18, 0.01), (0.21, 0.03), (0.27, 0.04), (0.32, 0.05), (0.40, 0.10) ] }
    means = [0.15, 0.15, 0.20]
    stds = [0.25, 0.27, 0.35]
    rhos = {(0, 1) : 0.1, (0, 2) : 0.1, (1, 2) : 0.1}
    lambdas = np.arange(0, 0.21, 0.01)
    ret = optimization.solve_mean_variance(lambdas, means, stds, rhos, fund_specs)