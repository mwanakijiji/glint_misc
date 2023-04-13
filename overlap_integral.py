#!/usr/bin/env python
# coding: utf-8

# Calculates overlap integral based on circular or hexagonal subapertures

import poppy
import pickle
import pandas as pd
import numpy as np
import astropy.modeling
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

## BEGIN USER INPUT
# read in complex fields from custom_PSF.py
file = open('./complex_fields_hex.pkl', 'rb')
# dump information to that file
amp, arg, I, aperture = pickle.load(file)
# close the file
file.close()
## END USER INPUT

# read in complex fields from circular aperture, for scaling
file = open('./complex_fields_circ.pkl', 'rb')
# dump information to that file
amp_circ, arg_circ, I_circ, aperture_circ = pickle.load(file)
# close the file
file.close()

# read in waveguide mode profile
'''
S. Gross:

The waveguide modes have a 4sigma diameter of 8.3x7.6um. 
A simple Gaussian fit gives a 1/e2 diameter of 5.8x5.4um. 
Both at a wavelength of 1550nm.

The attached CSV file contains the corresponding intensity profile. 
The scale is 0.2161um per pixel.
'''
stem = '/Users/bandari/Documents/git.repos/glint_misc/notebooks/data/'
open_file = open(stem + 'waveguide_intensity.pkl', "rb")
df_intensity, xycen = pickle.load(open_file)
open_file.close()

'''
plt.imshow(df_intensity, origin="lower")
plt.scatter(xycen[0],xycen[1], color='red', s=20)
plt.show()
'''

import ipdb; ipdb.set_trace()
# cutouts
buffer = 40 # pix
waveguide_cutout = df_intensity[int(xycen[1]-buffer):int(xycen[1]+buffer),int(xycen[0]-buffer):int(xycen[0]+buffer)]
cutout_airy = psf_circ[0].data[int(0.5*psf_circ[0].data.shape[0])-buffer:int(0.5*psf_circ[0].data.shape[0])+buffer,int(0.5*psf_circ[0].data.shape[1])-buffer:int(0.5*psf_circ[0].data.shape[1])+buffer]
cutout_hex = psf_hex[0].data[int(0.5*psf_hex[0].data.shape[0])-buffer:int(0.5*psf_hex[0].data.shape[0])+buffer,int(0.5*psf_hex[0].data.shape[1])-buffer:int(0.5*psf_hex[0].data.shape[1])+buffer]

# for secondary physical axes: 0.2161um per pixel.
def pix2um(x):
    return x*0.2161

def um2pix(x):
    return x/0.2161

# radius of first dark ring in um
wavel = 1.55 # um
foc_length = 400 # um
D = 66 # um

circ_r_um = 1.22 * wavel * foc_length/D
circ_r_pix = um2pix(circ_r_um)

# for checking scaling of illumination

'''
fig, ax = plt.subplots(layout='constrained')

ax.imshow(waveguide_cutout, extent=[-waveguide_cutout.shape[1]/2., waveguide_cutout.shape[1]/2., -waveguide_cutout.shape[0]/2., waveguide_cutout.shape[0]/2. ], origin='lower')
ax.set_xlabel('pixel')
ax.set_ylabel('pixel')

ax.imshow(cutout_airy, extent=[-waveguide_cutout.shape[1]/2., waveguide_cutout.shape[1]/2., -waveguide_cutout.shape[0]/2., waveguide_cutout.shape[0]/2. ], alpha=1, norm='log')

circ_cen_x = 0
circ_cen_y = 0 
circ = Circle((circ_cen_x,circ_cen_y),radius=circ_r_pix,color='white',fill=False)
ax.add_patch(circ)

secax = ax.secondary_xaxis('top', functions=(pix2um, um2pix))
secax.set_xlabel('physical (um)')
secay = ax.secondary_yaxis('right', functions=(pix2um, um2pix))
secay.set_ylabel('physical (um)')
plt.show()
'''

# define circle for scale in plots
circ_cen_x = 0
circ_cen_y = 0 
circ1 = Circle((circ_cen_x,circ_cen_y),radius=circ_r_pix,color='white',fill=False)
circ2 = Circle((circ_cen_x,circ_cen_y),radius=circ_r_pix,color='white',fill=False)
circ3 = Circle((circ_cen_x,circ_cen_y),radius=circ_r_pix,color='white',fill=False)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,10), layout='constrained')

ax[0,0].set_title('Circular subap')
ax[0,0].imshow(aper_circ_fits[0].data, cmap='Greys_r', interpolation='nearest')
ax[0,0].set_xlabel('(arbitrary)')
ax[0,0].set_ylabel('(arbitrary)')

ax[0,1].set_title('Hexagonal subap')
ax[0,1].imshow(aper_hex_fits[0].data, cmap='Greys_r', interpolation='nearest')
ax[0,1].set_xlabel('(arbitrary)')
#ax[0,0].set_tick_params(labelbottom=False)
#ax[0,0].set_tick_params(labelleft=False)

# Hide X and Y axes tick marks
#ax.set_xticks([])
#ax.set_yticks([])

# show normalized profiles of everything
ax[0,2].plot(np.divide(cutout_airy[int(buffer),:],np.max(cutout_airy[int(buffer),:])),color='red',label='circle')
ax[0,2].plot(np.divide(cutout_hex[int(buffer),:],np.max(cutout_hex[int(buffer),:])),color='blue',label='hexagon')
ax[0,2].plot(np.divide(waveguide_cutout[int(buffer),:],np.max(waveguide_cutout[int(buffer),:])),color='green',label='waveguide mode')
ax[0,2].legend()

ax[1,0].set_title('Illumination (linear)')
ax[1,0].imshow(cutout_airy, extent=[-waveguide_cutout.shape[1]/2., waveguide_cutout.shape[1]/2., -waveguide_cutout.shape[0]/2., waveguide_cutout.shape[0]/2. ], alpha=1)#, norm='log')
ax[1,0].set_xlabel('pixel')
ax[1,0].set_ylabel('pixel')
ax[1,0].add_patch(circ1)

#ax[1,1].imshow(waveguide_cutout, extent=[-waveguide_cutout.shape[1]/2., waveguide_cutout.shape[1]/2., -waveguide_cutout.shape[0]/2., waveguide_cutout.shape[0]/2. ], origin='lower')

ax[1,1].set_title('Illumination (linear)')
ax[1,1].imshow(cutout_hex, extent=[-waveguide_cutout.shape[1]/2., waveguide_cutout.shape[1]/2., -waveguide_cutout.shape[0]/2., waveguide_cutout.shape[0]/2. ], alpha=1)#, norm='log')
ax[1,1].set_xlabel('pixel')
ax[1,1].set_ylabel('pixel')
ax[1,1].add_patch(circ2)

ax[1,2].set_title('Waveguide mode (linear)')
ax[1,2].imshow(waveguide_cutout, extent=[-waveguide_cutout.shape[1]/2., waveguide_cutout.shape[1]/2., -waveguide_cutout.shape[0]/2., waveguide_cutout.shape[0]/2. ], alpha=1)
ax[1,2].set_xlabel('pixel')
ax[1,2].set_ylabel('pixel')
ax[1,2].add_patch(circ3)

secax = ax[1,0].secondary_xaxis('top', functions=(pix2um, um2pix))
secax.set_xlabel('physical (um)')
secay = ax[1,0].secondary_yaxis('right', functions=(pix2um, um2pix))
secay.set_ylabel('physical (um)')

secax = ax[1,1].secondary_xaxis('top', functions=(pix2um, um2pix))
secax.set_xlabel('physical (um)')
secay = ax[1,1].secondary_yaxis('right', functions=(pix2um, um2pix))
secay.set_ylabel('physical (um)')

secax = ax[1,2].secondary_xaxis('top', functions=(pix2um, um2pix))
secax.set_xlabel('physical (um)')
secay = ax[1,2].secondary_yaxis('right', functions=(pix2um, um2pix))
secay.set_ylabel('physical (um)')

secax = ax[0,2].secondary_xaxis('top', functions=(pix2um, um2pix))
secax.set_xlabel('physical (um)')
secay = ax[0,2].secondary_yaxis('right', functions=(pix2um, um2pix))
secay.set_ylabel('physical (um)')

plt.savefig('junk.pdf')


# make 'intensity' terms: E*E

I_airy = np.real(cutout_airy * np.conj(cutout_airy))

I_hex = np.real(cutout_hex * np.conj(cutout_hex))

I_waveguide = np.real(waveguide_cutout * np.conj(waveguide_cutout))

I_waveguide_x_airy = np.real(waveguide_cutout * np.conj(cutout_airy))

I_waveguide_x_hex = np.real(waveguide_cutout * np.conj(cutout_hex))

# eta_coeff_airy
# integrate by summing over all elements
numerator_airy = np.power(np.sum(I_waveguide_x_airy),2.)
denominator_airy = np.sum(I_airy) * np.sum(I_waveguide)
eta_coeff_airy = numerator_airy/denominator_airy

# eta_coeff_hex
numerator_hex = np.power(np.sum(I_waveguide_x_hex),2.)
denominator_hex = np.sum(I_hex) * np.sum(I_waveguide)
eta_coeff_hex = numerator_hex/denominator_hex

print('eta_coeff_airy:',eta_coeff_airy)
print('eta_coeff_hex:',eta_coeff_hex)