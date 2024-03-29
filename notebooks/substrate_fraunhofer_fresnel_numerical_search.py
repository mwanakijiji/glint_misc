#!/usr/bin/env python
# coding: utf-8

# Does numerical search over lens focal length, defocus, and compares subaperture shapes 
# to find best coupling

# Created 2023 Apr. 17 by E.S.

import poppy
import pickle
import pandas as pd
import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

stem = '/Users/bandari/Documents/git.repos/glint_misc/notebooks/data/'
#stem = '/suphys/espa3021/Documents/git.repos/glint_misc/notebooks/data/'

dir_plots = './junk/'

##############################################
## BEGIN USER-DEFINED QUANTITIES

# pixel scale of 'detector' image (plane of waveguide entrance)
scale_wg = 0.2161*u.micron/u.pix

# lenslet diameter
# BMC has pitch 647.5 um --> 0.9 ratio: 582.75 um
# IrisAO has pitch 606 um --> 0.9 ratio: 545.4 um
diam_lenslet = 58.275*u.micron # 66.*u.micron

# if true, secondary obscuration added in; which I think is bogus, 
# because subapertures should be sampled across the pupil in unobscured
# regions; ONLY USE TRUE IF YOU KNOW WHAT YOU'RE DOING
subaru_pupil = True

# smallest fractional distance between (projected) DM hexagonal subaperture
# and outer edge of subaperture that passes light
# (i.e., 0.1 for a ciruclar aperture means '0.9 of the radius of a circle out to the nearest boundary of 
# the DM subaperture hexagon passes light'; for a hexagonal aperture, 'the hexagon which passes light is 0.9
# the size of the DM subaperture)
stop_factor_wavefront = 0.1

# index of refraction of substrate
n_glass = 1.5255 # 1.5255 (D263), 1.5 (Gorilla)

## LOOP THIS
# wavelength in air (n = 1)
wavel_air = 1.55*u.micron

# focal lengths to test
grid_foc_length = range(300,600,20)

# steps in defocus (tenths of a wave)
grid_defocus = range(0,5,1)

'''
file_names_waveguide_modes = [stem + 'AF45_100x_Zeiss_300um_1550nm_1000mmmin_67pt5nJ_csv.pkl', 
                              stem + 'AF45_100x_Zeiss_300um_1550nm_1000mmmin_77pt5nJ_csv.pkl',
                              stem + 'AF45_100x_Zeiss_300um_1550nm_2000mmmin_67pt5nJ_csv.pkl',
                              stem + 'AF45_100x_Zeiss_300um_1550nm_500mmmin_75nJ_csv.pkl', 
                              stem + 'AF45_100x_Zeiss_300um_1550nm_750mmmin_50nJ_csv.pkl', 
                              stem + 'AF45_100x_Zeiss_300um_1550nm_750mmmin_77pt5nJ_csv.pkl']
'''
# to study just one waveguide
#file_names_waveguide_modes = [stem + 'AF45_100x_Zeiss_300um_1550nm_2000mmmin_67pt5nJ_csv.pkl']
file_names_waveguide_modes = [stem + 'AF45_100x_Zeiss_300um_1550nm_1000mmmin_67pt5nJ_csv.pkl']

## END USER-DEFINED QUANTITIES
##############################################

# for secondary physical axes: 0.2161um per pixel in MLA plane
def pix2um(x):
    # input is in units of pixels
    return x*0.2161*u.micron/u.pix

def um2pix(x):
    # input is in units of microns
    return (x/0.2161)*u.pix/u.micron

# derived quantities
# wavelength in substrate
wavel_substr = (wavel_air/n_glass)

# set up lists to hold all variables
list_f_lens = []
list_defocus = []
list_wavel = []
list_wg_modes = []
list_overlap_int_circ = []
list_overlap_int_hex = []
list_overlap_int_circ_fresnel = []
list_overlap_int_hex_fresnel = []

for file_name_wg in file_names_waveguide_modes:

    # read in waveguide mode

    '''
    S. Gross:

    The waveguide modes have a 4sigma diameter of 8.3x7.6um. 
    A simple Gaussian fit gives a 1/e2 diameter of 5.8x5.4um. 
    Both at a wavelength of 1550nm.

    The attached CSV file contains the corresponding intensity profile. 
    The scale is 0.2161um per pixel.
    '''

    # retrieve waveguide intensity and make cutout

    open_file = open(file_name_wg, "rb")
    df_intensity, xycen = pickle.load(open_file)
    open_file.close()
    # cutouts
    buffer = 100 # pix
    waveguide_cutout = df_intensity[int(xycen[1]-buffer):int(xycen[1]+buffer),int(xycen[0]-buffer):int(xycen[0]+buffer)]


    # ## Set up optics and calculate PSF (focal length implicit in optical system)

    for foc_length in grid_foc_length:

        # initial focal length of lens (um) in the substrate
        f_lens_substr = foc_length*u.micron # 284.664*u.micron
        
        # radius of first dark ring in um
        circ_r_um = 1.22 * wavel_substr * f_lens_substr/diam_lenslet
        circ_r_pix = um2pix(circ_r_um)
        circ_r_asec = (circ_r_um/f_lens_substr)*206265.*u.arcsec
        
        # pixel scale in terms of arcsec/pix
        pixelscale_ang = (scale_wg/f_lens_substr)*206265.*u.arcsec # (0.2161um/f [um] )*206265 arcsec /pix

        steps_one_side = 4 # number of steps on one side of zero focus
        overl_int_array = np.nan*np.ones(int(2*steps_one_side)) # will collect overlap integrals
        defocus_values_array = np.nan*np.ones(int(2*steps_one_side)) # will collect defocus values (ito waves)

        for step_foc in grid_defocus:

            NWAVES = step_foc*0.1 # size of defocus ito waves
            #import ipdb; ipdb.set_trace()
            ###############################
            # construct system: circular, Fraunhofer
            osys_circ = poppy.OpticalSystem()
            # lenslet (the factor of /(1.-stop_factor_wavefront) in the radius stops 'up' the incoming wavefront by the stop_factor_wavefront;
            # note this does not change the size of the lenslet)
            lens = osys_circ.add_pupil(optic = poppy.ThinLens(name='lenslet', nwaves=NWAVES, radius=0.5*diam_lenslet/(1.-stop_factor_wavefront)))
            # impose circular aperture (redundant here, but need to be consistent with hexagonal aperture)
            # filled-aperture
            circ_ap = osys_circ.add_pupil(poppy.CircularAperture(radius=0.5*diam_lenslet))
            # symmetric secondary obscuration
            #sec_obsc = osys_circ.add_pupil(poppy.SecondaryObscuration(secondary_radius=0.1*diam_lenslet, support_width=0.000001))
            if subaru_pupil:
                # Subaru secondary obscuration (parameters by-eye)
                sec_obsc = osys_circ.add_pupil(poppy.AsymmetricSecondaryObscuration(secondary_radius=0.14*diam_lenslet,
                                    support_angle=(40, 140, 220, 320),
                                    support_width=np.multiply(1e-6,[1, 1, 1, 1]),
                                    support_offset_x=np.multiply(4.5e-6,[-1, -1, 1, 1]),
                                    name='Complex secondary'))
            # final focal plane
            det = osys_circ.add_detector(pixelscale=pixelscale_ang, fov_pixels=200, oversample=1)  # note oversample=1 to make pixel scale consistent
            # calculate
            psf_circ, all_wfs_circ = osys_circ.calc_psf(wavelength=wavel_substr, 
                                    display_intermediates=False,
                                    return_intermediates=True)
            #import ipdb; ipdb.set_trace()
            ###############################
            # construct system: hexagonal, Fraunhofer
            osys_hex = poppy.OpticalSystem()
            # lenslet (the factor of /(1.-stop_factor_wavefront) in the radius stops 'up' the incoming wavefront by the stop_factor_wavefront)
            lens = osys_hex.add_pupil(optic = poppy.ThinLens(name='lenslet', nwaves=NWAVES, radius=0.5*diam_lenslet/(1.-stop_factor_wavefront)))
            # impose hexagonal aperture
            hex_ap = osys_hex.add_pupil(optic = poppy.HexagonAperture(flattoflat=diam_lenslet), name='hex')
            if subaru_pupil:
                # Subaru secondary obscuration (parameters by-eye)
                sec_obsc = osys_hex.add_pupil(poppy.AsymmetricSecondaryObscuration(secondary_radius=0.14*diam_lenslet,
                                    support_angle=(40, 140, 220, 320),
                                    support_width=np.multiply(1e-6,[1, 1, 1, 1]),
                                    support_offset_x=np.multiply(4.5e-6,[-1, -1, 1, 1]),
                                    name='Complex secondary'))
            # final focal plane
            det = osys_hex.add_detector(pixelscale=pixelscale_ang, fov_pixels=200, oversample=1)  # note oversample=1 to make pixel scale consistent
            # calculate
            psf_hex, all_wfs_hex = osys_hex.calc_psf(wavelength=wavel_substr, 
                                    display_intermediates=False,
                                    return_intermediates=True)
            
            ###############################
            # construct system: circular, Fresnel
            opt_sys_circ_fresnel = poppy.FresnelOpticalSystem(pupil_diameter=diam_lenslet, npix=512, beam_ratio=0.25)
            ap = poppy.CircularAperture(radius=0.5*diam_lenslet)
            m1 = poppy.QuadraticLens(f_lens=f_lens_substr, name='lenslet')

            opt_sys_circ_fresnel.add_optic(ap, distance=0*u.micron)
            opt_sys_circ_fresnel.add_optic(m1, distance=0*u.micron)
            opt_sys_circ_fresnel.add_detector(pixelscale=scale_wg, fov_pixels=200, distance=f_lens_substr)
            psf_circ_fresnel, all_wfs_circ_fresnel = opt_sys_circ_fresnel.calc_psf(wavelength=wavel_substr, 
                                                       display_intermediates=False, 
                                                       return_intermediates=True)
            
            ###############################
            # construct system: hexagonal, Fresnel
            opt_sys_hex_fresnel = poppy.FresnelOpticalSystem(pupil_diameter=diam_lenslet, npix=512, beam_ratio=0.25)
            ap = poppy.HexagonAperture(flattoflat=diam_lenslet, name='hex')
            m1 = poppy.QuadraticLens(f_lens=f_lens_substr, name='lenslet')

            opt_sys_hex_fresnel.add_optic(ap, distance=0*u.micron)
            opt_sys_hex_fresnel.add_optic(m1, distance=0*u.micron)
            opt_sys_hex_fresnel.add_detector(pixelscale=scale_wg, fov_pixels=200, distance=f_lens_substr)
            psf_hex_fresnel, all_wfs_hex_fresnel = opt_sys_hex_fresnel.calc_psf(wavelength=wavel_substr, 
                                                       display_intermediates=False, 
                                                       return_intermediates=True)
        
            
            #print('shape:',np.shape(all_wfs_circ[-1].intensity))


            ###############################
            # define fields for overlap integral (should just be 200x200 arrays)
            input_field_circ = all_wfs_circ[-1].amplitude * np.exp(1j*all_wfs_circ[-1].phase)
            input_field_hex = all_wfs_hex[-1].amplitude * np.exp(1j*all_wfs_hex[-1].phase)

            input_field_circ_fresnel = all_wfs_circ_fresnel[-1].amplitude * np.exp(1j*all_wfs_circ_fresnel[-1].phase)
            input_field_hex_fresnel = all_wfs_hex_fresnel[-1].amplitude * np.exp(1j*all_wfs_hex_fresnel[-1].phase)

            mode_field = np.sqrt(waveguide_cutout) # sqrt because cutout is the intensity I, and we want E

            # check scaling is right
            #plt.imshow(all_wfs[-1].intensity,norm='log')
            #plt.show()

            # overlap integral
            overlap_int_complex_circ = np.sum(input_field_circ*mode_field) / np.sqrt( np.sum(np.abs(mode_field)**2) * np.sum(np.abs(input_field_circ)**2) )
            overlap_int_complex_hex = np.sum(input_field_hex*mode_field) / np.sqrt( np.sum(np.abs(mode_field)**2) * np.sum(np.abs(input_field_hex)**2) )
            overlap_int_complex_circ_fresnel = np.sum(input_field_circ_fresnel*mode_field) / np.sqrt( np.sum(np.abs(mode_field)**2) * np.sum(np.abs(input_field_circ_fresnel)**2) )
            overlap_int_complex_hex_fresnel = np.sum(input_field_hex_fresnel*mode_field) / np.sqrt( np.sum(np.abs(mode_field)**2) * np.sum(np.abs(input_field_hex_fresnel)**2) )


            overlap_int_circ = np.abs(overlap_int_complex_circ)**2
            overlap_int_hex = np.abs(overlap_int_complex_hex)**2
            overlap_int_circ_fresnel = np.abs(overlap_int_complex_circ_fresnel)**2
            overlap_int_hex_fresnel = np.abs(overlap_int_complex_hex_fresnel)**2

            # print params
            string_params = 'Waveguide mode: '+str(os.path.basename(file_name_wg))+'\n' + \
                            'MLA lenslet focal length in substrate (um): '+str(f_lens_substr)+'\n' + \
                            'MLA lenslet diameter (um): '+str(diam_lenslet)+'\n' + \
                            'Defocus (waves): '+str(NWAVES)+'\n' + \
                            'Wavel in air (um): '+str(wavel_air)+'\n' + \
                            'Wavel in substrate (um): '+str(wavel_substr)+'\n' + \
                            'Substrate index of refraction: '+str(n_glass)+'\n' + \
                            'Radius of first dark Airy ring (asec): '+str(circ_r_asec)+'\n' + \
                            'Radius of first dark Airy ring (um): '+str(circ_r_um)+'\n' + \
                            'Radius of first dark Airy ring (pix): '+str(circ_r_pix)+'\n' + \
                            '(white circle: first dark Airy ring, as sanity check)'+'\n' + \
                            '---------------------'+'\n' + \
                            'OVERLAP INTEGRAL (circ, Fraun): ' + str(overlap_int_circ)+'\n' + \
                            'OVERLAP INTEGRAL (hex, Fraun): ' + str(overlap_int_hex)+'\n' + \
                            'OVERLAP INTEGRAL (circ, Fresnel): ' + str(overlap_int_circ_fresnel)+'\n' + \
                            'OVERLAP INTEGRAL (hex, Fresnel): ' + str(overlap_int_hex_fresnel)+'\n' + \
                            '---------------------'
                    
                
            print('######################')
            print(string_params)

            # append everything
            list_f_lens.append(f_lens_substr.value)
            list_defocus.append(NWAVES)
            list_wavel.append(wavel_substr.value)
            list_wg_modes.append(os.path.basename(file_name_wg))
            list_overlap_int_circ.append(overlap_int_circ)
            list_overlap_int_hex.append(overlap_int_hex)
            list_overlap_int_circ_fresnel.append(overlap_int_circ_fresnel)
            list_overlap_int_hex_fresnel.append(overlap_int_hex_fresnel)

            #import ipdb; ipdb.set_trace()
            #########################
            ## plotting
            
            # intensities for plotting
            I_PSF_circ = all_wfs_circ[-1].intensity
            I_PSF_hex = all_wfs_hex[-1].intensity
            I_PSF_circ_fresnel = all_wfs_circ_fresnel[-1].intensity
            I_PSF_hex_fresnel = all_wfs_hex_fresnel[-1].intensity
            
            # define circle for scale in plots
            circ_cen_x = 0
            circ_cen_y = 0 
            circ1 = Circle((circ_cen_x,circ_cen_y),radius=circ_r_pix/u.pix,color='white',fill=False)
            circ2 = Circle((circ_cen_x,circ_cen_y),radius=circ_r_pix/u.pix,color='white',fill=False)
            circ3 = Circle((circ_cen_x,circ_cen_y),radius=circ_r_pix/u.pix,color='white',fill=False)
            circ4 = Circle((circ_cen_x,circ_cen_y),radius=circ_r_pix/u.pix,color='white',fill=False)
            circ5 = Circle((circ_cen_x,circ_cen_y),radius=circ_r_pix/u.pix,color='white',fill=False)
            
            plt.clf()
            fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(30,10), layout='constrained')
            #fig.tight_layout()

            plt.suptitle(string_params, x=0.1, horizontalalignment='left')
            ax[0,0].set_title('Circular subap')
            ax[0,0].imshow(all_wfs_circ[-2].amplitude, cmap='Greys_r', interpolation='nearest')
            ax[0,0].set_xlabel('(arbitrary)')
            ax[0,0].set_ylabel('(arbitrary)')

            ax[0,1].set_title('Hexagonal subap')
            ax[0,1].imshow(all_wfs_hex[-2].amplitude, cmap='Greys_r', interpolation='nearest')
            ax[0,1].set_xlabel('(arbitrary)')
            ax[0,1].set_ylabel('(arbitrary)')

            # show normalized profiles of everything
            ax[0,2].plot(np.divide(I_PSF_circ[int(buffer),:],np.max(I_PSF_circ[int(buffer),:])),color='red',label='circle')
            ax[0,2].plot(np.divide(I_PSF_hex[int(buffer),:],np.max(I_PSF_hex[int(buffer),:])),color='blue',label='hexagon')
            ax[0,2].plot(np.divide(waveguide_cutout[int(buffer),:],np.max(waveguide_cutout[int(buffer),:])),color='green',label='waveguide mode')
            ax[0,2].legend()

            ax[1,0].set_title('I, Fraunhofer (log)')
            ax[1,0].imshow(I_PSF_circ, extent=[-I_PSF_circ.shape[1]/2., I_PSF_circ.shape[1]/2., -I_PSF_circ.shape[0]/2., I_PSF_circ.shape[0]/2. ], alpha=1, norm='log')
            ax[1,0].set_xlabel('pixel')
            ax[1,0].set_ylabel('pixel')
            ax[1,0].add_patch(circ1)


            ax[1,1].set_title('I, Fraunhofer (log)')
            ax[1,1].imshow(I_PSF_hex, extent=[-I_PSF_hex.shape[1]/2., I_PSF_hex.shape[1]/2., -I_PSF_hex.shape[0]/2., I_PSF_hex.shape[0]/2. ], alpha=1, norm='log')
            ax[1,1].set_xlabel('pixel')
            ax[1,1].set_ylabel('pixel')
            ax[1,1].add_patch(circ2)

            ax[1,2].set_title('Waveguide mode I (linear)')
            ax[1,2].imshow(waveguide_cutout, extent=[-waveguide_cutout.shape[1]/2., waveguide_cutout.shape[1]/2., -waveguide_cutout.shape[0]/2., waveguide_cutout.shape[0]/2. ], alpha=1)
            ax[1,2].set_xlabel('pixel')
            ax[1,2].set_ylabel('pixel')
            ax[1,2].add_patch(circ3)

            ax[2,0].set_title('I, Fresnel (log)')
            ax[2,0].imshow(I_PSF_circ_fresnel, extent=[-I_PSF_circ_fresnel.shape[1]/2., I_PSF_circ_fresnel.shape[1]/2., -I_PSF_circ_fresnel.shape[0]/2., I_PSF_circ_fresnel.shape[0]/2. ], alpha=1, norm='log')
            ax[2,0].set_xlabel('pixel')
            ax[2,0].set_ylabel('pixel')
            ax[2,0].add_patch(circ4)


            ax[2,1].set_title('I, Fresnel (log)')
            ax[2,1].imshow(I_PSF_hex_fresnel, extent=[-I_PSF_hex_fresnel.shape[1]/2., I_PSF_hex_fresnel.shape[1]/2., -I_PSF_hex_fresnel.shape[0]/2., I_PSF_hex_fresnel.shape[0]/2. ], alpha=1, norm='log')
            ax[2,1].set_xlabel('pixel')
            ax[2,1].set_ylabel('pixel')
            ax[2,1].add_patch(circ5)

            ax[2,2].axis('off')

            '''
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
            '''

            fyi_plot_filename = dir_plots+'foc_'+str(f_lens_substr.value)+'_'+\
                        'defoc_waves_'+str(NWAVES)+'_'+\
                        'wavel_' + str(wavel_substr.value) + '_'+\
                        'wg_' + str(os.path.basename(file_name_wg))+ '.png'
            plt.savefig(fyi_plot_filename)
            plt.close()
            print('Wrote',fyi_plot_filename)

        # save interim data
        # put everything into a final dataframe
        df_interim = pd.DataFrame(list(zip(list_f_lens,list_defocus,list_wavel,list_wg_modes,list_overlap_int_circ,list_overlap_int_hex,list_overlap_int_circ_fresnel,list_overlap_int_hex_fresnel)),
                                  columns=['f_lens_substrate '+str(f_lens_substr.unit),'defocus (waves)','wavel '+str(wavel_substr.unit),'wg_mode','overl_int_circ_fraun','overlap_int_hex_fraun','overl_int_circ_fresnel','overlap_int_hex_fresnel'])
        df_interim.to_csv('junk_df_interim.csv', sep=',', index=False)

# put everything into a final dataframe
df_final = pd.DataFrame(list(zip(list_f_lens,list_defocus,list_wavel,list_wg_modes,list_overlap_int_circ,list_overlap_int_hex,list_overlap_int_circ_fresnel,list_overlap_int_hex_fresnel)),
                                  columns=['f_lens_substrate '+str(f_lens_substr.unit),'defocus (waves)','wavel '+str(wavel_substr.unit),'wg_mode','overl_int_circ_fraun','overlap_int_hex_fraun','overl_int_circ_fresnel','overlap_int_hex_fresnel'])

df_final.to_csv('junk_final.csv', sep=',', index=False)
#import ipdb; ipdb.set_trace()
'''
plt.plot(defocus_values_array,overl_int_array)
plt.axvline(x=0,linestyle=':',color='k')
plt.xlabel('Defocus/$\lambda$')
plt.ylabel('$\eta$')
plt.savefig('junk.png')
'''

'''
plt.plot(list_f_lens,list_overlap_int_circ,label='Circ')
plt.plot(list_f_lens,list_overlap_int_hex,label='Hex')
plt.legend()
plt.show()
'''

# example code from BN

'''
psf, all_wfs = osys.calc_psf(wavelength=WAVELENGTH, display_intermediates=False,
                                             return_intermediates=True)
pupil_wf = all_wfs[2]
pupil_ampl = np.abs(pupil_wf.wavefront)
pupil_phase = pupil_wf.phase # One way to get phase

final_wf = all_wfs[-1]
psf_ampl = np.abs(final_wf.wavefront)
psf_phase = np.angle(final_wf.wavefront) # Aother way to get phase
'''