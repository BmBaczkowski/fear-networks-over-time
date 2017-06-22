# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 21:07:19 2016

@author: baczkowski
"""

def calc_trends(nr_vols):

    import numpy as np
    import os
    
    linear = np.arange(0, nr_vols)
    quadratic = np.arange(0, nr_vols)**2
    trends = np.column_stack((linear,quadratic))

    out_file = os.path.join(os.getcwd(), 'linear_quadratic_trends.txt')
    np.savetxt(out_file, trends, fmt='%d', delimiter='\t')
    
    return out_file 
    