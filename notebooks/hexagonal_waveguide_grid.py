#!/usr/bin/env python
# coding: utf-8

# Computes the coordinates of waveguide inputs for the GLINT chip

from hexalattice.hexalattice import *

# pitches of DMs

pitch_irisAO = 606. # um
pitch_BMC = 647.5 # um

# magnification DM -> MLA
M = 0.1

hex_centers_irisAO, _ = create_hex_grid(nx=4,
                                 ny=4,
                                 min_diam=M*pitch_irisAO,align_to_origin=False,
                                 do_plot=False)
                                 
hex_centers_BMC, _ = create_hex_grid(nx=4,
                                 ny=4,
                                 min_diam=M*pitch_BMC,align_to_origin=False,
                                 do_plot=False)


centers_x_irisAO = hex_centers_irisAO[:, 0]
centers_y_irisAO = hex_centers_irisAO[:, 1]

centers_x_BMC = hex_centers_BMC[:, 0]
centers_y_BMC = hex_centers_BMC[:, 1]

xtick_loc = 20*np.arange(13)
ytick_loc = 20*np.arange(10)

plt.clf()
plt.scatter(centers_x_irisAO,centers_y_irisAO)
plt.title('Waveguide centers (IrisAO)')
plt.xlabel('um')
plt.ylabel('um')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.set_xticks(xtick_loc)
ax.set_yticks(ytick_loc)
plt.grid(visible=True,which='major')
plt.savefig('iris.png')

xtick_loc = 20*np.arange(13)
ytick_loc = 20*np.arange(10)

plt.clf()
plt.scatter(centers_x_BMC,centers_y_BMC)
plt.title('Waveguide centers (BMC)')
plt.xlabel('um')
plt.ylabel('um')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.set_xticks(xtick_loc)
ax.set_yticks(ytick_loc)
plt.grid(visible=True,which='major')
plt.savefig('bmc.png')

print('Iris AO grid:')
print(np.column_stack((centers_x_irisAO,centers_y_irisAO)))

print('------------')
print('BMC grid:')
print(np.column_stack((centers_x_BMC,centers_y_BMC)))

