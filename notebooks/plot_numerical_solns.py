#!/usr/bin/env python
# coding: utf-8

# Plots numerical solutions to substrate thickness

# Created 2023 May 1 by E.S.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stem = '/Users/bandari/Documents/git.repos/glint_misc/notebooks/'

df = pd.read_csv(stem + 'numerical_optimization_1st_order.csv')

# separate by waveguide mode
wg_unique_list = df['wg_mode'].drop_duplicates().values
# separate by defocus
defocus_unique_list = df['defocus (waves)'].drop_duplicates().values

for wg_this in wg_unique_list:
    for defocus_this in defocus_unique_list:
        
        idx = np.logical_and(df['wg_mode'] == wg_this, df['defocus (waves)'] == defocus_this)
    
        df_this_combo = df.loc[idx]
    
        plt.plot(df_this_combo['f_lens_substrate micron'],df_this_combo['overl_int_circ'], color='blue')
        plt.plot(df_this_combo['f_lens_substrate micron'],df_this_combo['overlap_int_hex'], color='red')
        
        idx_max = df_this_combo['overl_int_circ'] == np.max(df_this_combo['overl_int_circ'])
        
        print(defocus_this)
        if defocus_this < 0.1:
            plt.annotate( str(defocus_this), (df_this_combo['f_lens_substrate micron'].loc[idx_max], df_this_combo['overl_int_circ'].loc[idx_max]) )

    plt.axvline(x=400.,linestyle='--',color='k')
    
    plt.xlabel('lenslet focal length in substrate (um)')
    plt.ylabel('overlap integral')
    plt.legend()
    plt.title(wg_this + '\n' + 'blue: circular; red: hexagonal' + '\n' + 'numbers: defocus [waves]')
    plt.savefig('junk.png')
    plt.clf()