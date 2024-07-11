# Reads in a large VAMPIRES data cube, and saves the first slice as a separate frame
# (meant to do this in scexao6, so that I can do some analysis locally)

# Created 2024 July 12 by E.S.

import glob
from astropy.io import fits
import numpy as np
import os

stem = '/Volumes/One Touch/20240509_apapane_fake_seeing_spectra/'

fits_files = glob.glob(stem + '*.fits')

data = []
for file_name in fits_files:
    hdul = fits.open(file_name)

    hdul.info()

    image = hdul[0].data[0,:,:]

    file_name_write = stem + 'single_slice_' + os.path.basename(file_name)
    fits.writeto(file_name_write, image.astype(np.float32), overwrite=True)
    print('Wrote',file_name_write)


