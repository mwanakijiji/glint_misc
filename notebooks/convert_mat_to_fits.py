import scipy.io
import h5py
import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

stem = '/Volumes/One Touch/glint_old_data/alfBoo_1/'

file_list = glob.glob(stem + '*.mat')

# Open the HDF5 file
for file_name in file_list:
    with h5py.File(file_name, 'r') as file:

        # Check if 'imagedata' exists in the file
        if 'imagedata' in file:
            imagedata = file['imagedata']
            imagedata_array = imagedata[:]

        else:
            print("Dataset 'imagedata' not found in the file.")

        # Write out imagedata_array as a FITS file
        file_name_write = file_name.split('.')[0] + '.fits'
        hdu = fits.PrimaryHDU(imagedata_array.astype(imagedata_array.dtype))
        hdu.writeto(file_name_write, overwrite=True)
        print("FITS file saved as:", file_name_write)