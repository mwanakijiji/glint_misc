import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

stem = '/Users/eckhartspalding/Documents/git.repos/glint_misc/notebooks/data/'

mat_data = loadmat(stem + 'glint_waveguide_plot.mat')

wg_names = ['N1', 'N2', 'N3', 'B1B', 'B3A', '', 'P1', '', 'B2B', 'B3B', '', 'P3', '', 'B1A', 'B2A', '', 'P2', '' ]

# Define a list of 18 distinct colors
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',  # Default matplotlib colors
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',  # More matplotlib colors
    '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d'   # Additional distinct colors
]

plt.clf()

# Create a figure with three subplots in one row
plt.figure(figsize=(15, 10))  # Made figure taller to accommodate two rows

# First row - Not to scale
# YX projection
plt.subplot(231)
for wg_num in range(len(mat_data['data'][:][0]['X'])):
    valid_mask = np.isfinite(mat_data['data'][0][wg_num]['X']) & np.isfinite(mat_data['data'][0][wg_num]['Y'])
    plt.plot(mat_data['data'][0][wg_num]['Y'][valid_mask], 
             mat_data['data'][0][wg_num]['X'][valid_mask], 
             label=str(wg_names[wg_num]), 
             linewidth=2,
             color=colors[wg_num])
plt.xlabel('Y')
plt.ylabel('X')
plt.title('YX Projection (Auto-scaled)')
plt.legend()
plt.grid()

# XZ projection
plt.subplot(232)
for wg_num in range(len(mat_data['data'][:][0]['X'])):
    valid_mask = np.isfinite(mat_data['data'][0][wg_num]['X']) & np.isfinite(mat_data['data'][0][wg_num]['Z'])
    plt.plot(mat_data['data'][0][wg_num]['X'][valid_mask], 
             mat_data['data'][0][wg_num]['Z'][valid_mask], 
             label=str(wg_names[wg_num]), 
             linewidth=2,
             color=colors[wg_num])
plt.xlabel('X')
plt.ylabel('Z')
plt.title('XZ Projection (Auto-scaled)')
plt.grid()

# YZ projection
plt.subplot(233)
for wg_num in range(len(mat_data['data'][:][0]['X'])):
    valid_mask = np.isfinite(mat_data['data'][0][wg_num]['Y']) & np.isfinite(mat_data['data'][0][wg_num]['Z'])
    plt.plot(mat_data['data'][0][wg_num]['Y'][valid_mask], 
             mat_data['data'][0][wg_num]['Z'][valid_mask], 
             label=str(wg_names[wg_num]), 
             linewidth=2,
             color=colors[wg_num])
plt.xlabel('Y')
plt.ylabel('Z')
plt.title('YZ Projection (Auto-scaled)')
plt.grid()

def get_axis_limits(data, dim1, dim2):
    min1, max1 = float('inf'), float('-inf')
    min2, max2 = float('inf'), float('-inf')
    
    for wg_num in range(len(data['data'][0])):
        valid_mask = np.isfinite(data['data'][0][wg_num][dim1]) & np.isfinite(data['data'][0][wg_num][dim2])
        if np.any(valid_mask):
            min1 = min(min1, np.min(data['data'][0][wg_num][dim1][valid_mask]))
            max1 = max(max1, np.max(data['data'][0][wg_num][dim1][valid_mask]))
            min2 = min(min2, np.min(data['data'][0][wg_num][dim2][valid_mask]))
            max2 = max(max2, np.max(data['data'][0][wg_num][dim2][valid_mask]))
    
    # Add 5% padding
    range1 = max1 - min1
    range2 = max2 - min2
    padding1 = range1 * 0.05
    padding2 = range2 * 0.05
    
    return [min1 - padding1, max1 + padding1, min2 - padding2, max2 + padding2]

# Bottom row - Single wide plot to scale
ax = plt.subplot(212)  # 2 rows, 1 column, plot 2
for wg_num in range(len(mat_data['data'][:][0]['X'])):
    valid_mask = np.isfinite(mat_data['data'][0][wg_num]['X']) & np.isfinite(mat_data['data'][0][wg_num]['Y'])
    ax.plot(mat_data['data'][0][wg_num]['Y'][valid_mask], 
             mat_data['data'][0][wg_num]['X'][valid_mask], 
             label=str(wg_names[wg_num]), 
             linewidth=2,
             color=colors[wg_num])
ax.set_aspect('equal')
ax.set_xlim(0, 65)
ax.set_ylim(-8, 2)

plt.tight_layout()
plt.show()


