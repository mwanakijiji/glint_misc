import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Circle

#Create Fizeau PSF

phase=np.pi/2.0*0.0
D=20000 #mm # 8250 # scale the output array size with this; 10000 for 100x100 psf cutout
padfactor=20.0 # 16 # scale the PSF with this
deltaspacing = 100 #mm # 100
centerx=D*padfactor/deltaspacing/2.0 #pixels
centery=D*padfactor/deltaspacing/2.0 #pixels
hwstamp = D/2.0/deltaspacing #pixels # THIS SETS CHARACTERISTIC SCALE OF IMAGE ON PIXEL PLANE

def LBT_field(phase):
    # returns the physical aperture, and the complex aperture*phase

    lenx = D*padfactor/deltaspacing #pixels
    leny = D*padfactor/deltaspacing #pixels

    xpos = np.arange(-D/2.0*padfactor,D/2.0*padfactor,deltaspacing)
    ypos = np.arange(-D/2.0*padfactor,D/2.0*padfactor,deltaspacing)
    xx,yy = np.meshgrid(xpos,ypos)
    xsx = xx - 7208.5
    xdx = xx + 7208.5
    rr = np.sqrt(xx**2+yy**2)
    rrsx = np.sqrt(xsx**2+yy**2)
    rrdx = np.sqrt(xdx**2+yy**2)
    sxAperture = rrsx<D/2.0
    sxAperturephase = sxAperture * 1.0
    dxAperture = rrdx<D/2.0
    dxAperturephase = dxAperture * np.exp(1j*phase)
    LBTAperture = sxAperture + dxAperture
    LBTAperturephase = sxAperturephase + dxAperturephase

    return LBTAperture, LBTAperturephase


def circ_field(phase):
    # returns the physical aperture, and the complex aperture*phase

    # dimensions of entire array
    lenx = D*padfactor/deltaspacing #pixels
    leny = D*padfactor/deltaspacing #pixels

    xpos = np.arange(-D/2.0*padfactor,D/2.0*padfactor,deltaspacing)
    ypos = np.arange(-D/2.0*padfactor,D/2.0*padfactor,deltaspacing)
    xx, yy = np.meshgrid(xpos,ypos)
    xdx = xx
    rr = np.sqrt(xx**2+yy**2)
    rrdx = np.sqrt(xdx**2+yy**2)
    dxAperture = rrdx<D/2.0
    dxAperturephase = dxAperture * np.exp(1j*phase)
    aperture = dxAperture
    aperturephase = dxAperturephase

    return aperture, aperturephase


def hex_field(phase):
    # returns the physical aperture, and the complex aperture*phase

    # load the hexagonal shape
    file = open('./notebooks/hex_aperture.pkl', 'rb')
    # dump information to that file
    hex_shape = pickle.load(file)
    # close the file
    file.close()

    # this shape will be scaled so that the flat-to-flat edge (along y-direction) length is 
    # D (which is equiv to 2.*hwstam on pixel plane)
    y_original, x_original = np.shape(hex_shape)
    # rescale the hexagon
    y_cutout_new = int(2*hwstamp) # flat-to-flat edge (in pixel space)
    x_cutout_new = int( x_original * (2*hwstamp) / y_original )

    hex_shape_scaled = cv2.resize(hex_shape, dsize=(x_cutout_new, y_cutout_new), interpolation=cv2.INTER_LINEAR)
    hex_shape_scaled_bool = np.round(hex_shape_scaled) # forces edge pixels to be 0 or 1

    lenx = D*padfactor/deltaspacing #pixels
    leny = D*padfactor/deltaspacing #pixels

    array_initial = np.zeros((int(lenx),int(leny)))
    # replace center of array with hexagonal aperture
    y_cutout, x_cutout = np.shape(hex_shape_scaled_bool)
    array_initial[int(0.5*(leny-y_cutout)):int(0.5*(leny+y_cutout)), int(0.5*(lenx-x_cutout)):int(0.5*(lenx+x_cutout))] = hex_shape_scaled_bool

    aperture = array_initial
    aperturephase = array_initial * np.exp(1j*phase)

    return aperture, aperturephase


def gen_PSF(aperture_phase_choice):
    # turn complex field at the aperture into a PSF

    #aperture_phase_choice = hex_aperture_phase
    #aperture_simple_choice = hex_aperture


    # [ complex aperture ]  <---autocorrel--->  [ OTF ]
    #          |                                    |
    #          FT                                   FT
    #          |                                    |
    #        [ A ]          <---|  |^2----->    [ PSF ]
    #
    #  N.b.: [ A ] is the complex field AT THE FOCAL PLANE (modulo a quadratic phase factor; 
    # see Tyson, sec. 4.2.1, 'Principles and Applications of Fourier Optics')

    # create [ A ]
    FTAp = np.fft.fft2(aperture_phase_choice) # 2D FT
    FTAp = np.fft.fftshift(FTAp) # shift zero freq to center
    FTAp = FTAp[int(centerx-hwstamp):int(centerx+hwstamp),int(centery-hwstamp):int(centery+hwstamp)]

    # kludge to remove checkerboard from real and imaginary terms in [ A ] which will be returned to user
    FTAp_return_big = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(aperture_phase_choice)))
    buffer = 100
    y_big, x_big = np.shape(FTAp_return_big)
    FTAp_return = FTAp_return_big[int(0.5*y_big)-buffer:int(0.5*y_big)+buffer,int(0.5*x_big)-buffer:int(0.5*x_big)+buffer] # return cutout (NOTE THE ARBITRARY HARD-CODED SIZE!)

    # create [ PSF ]
    I = np.real(FTAp * np.conj(FTAp))

    #Now extract results from the PSF  
    cutI=I[0:len(I),0:len(I)-4]   #getting the software aperture offset by a certain amount

    padI=np.pad(cutI,len(cutI),'edge')

    # create [ OTF ]
    #This was needed to get rid of the checkerboard pattern
    padI = np.fft.fftshift(padI) # shift zero freq to center

    PhaseExtract = np.fft.fft2(padI) # 2D FT
    PhaseExtract = np.fft.fftshift(PhaseExtract)
    # amplitude and angle of OTF
    AmpPE = np.absolute(PhaseExtract)
    ArgPE = np.angle(PhaseExtract)

    CenterPix = len(PhaseExtract)/2.0
    IntPix = 150.0

    CenterPhase = ArgPE[int(CenterPix),int(CenterPix)]
    InterfPhase = ArgPE[int(CenterPix),int(IntPix)] * 180.0/np.pi

    print("Phase of Interference fringes = ",   InterfPhase)

    # return the complex field at focal plane [ A ], and the resulting PSF intensity [ PSF ], OTF amplitude, OTF argument
    return FTAp_return, I, AmpPE, ArgPE


# for secondary physical axes: 0.2161um per pixel.
def pix2um(x):
    return x*0.2161

def um2pix(x):
    return x/0.2161


def main():

    #LBTAperture, LBTAperturephase = LBT_field(phase)
    circ_aperture_simple, circ_aperture_phase = circ_field(phase)
    hex_aperture_simple, hex_aperture_phase = hex_field(phase)

    ### USER INPUT
    #aperture_simple_choice = circ_aperture
    #aperture_phase_choice = circ_aperture_phase
    ### END USER INPUT

    # from circular subap
    field_complex_A_circ, I_PSF_circ, amp_OTF_circ, arg_OTF_circ = gen_PSF(circ_aperture_phase)
    # from hexagonal subap
    field_complex_A_hex, I_PSF_hex, amp_OTF_hex, arg_OTF_hex = gen_PSF(hex_aperture_phase)

    '''
    S. Gross:

    The waveguide modes have a 4sigma diameter of 8.3x7.6um. 
    A simple Gaussian fit gives a 1/e2 diameter of 5.8x5.4um. 
    Both at a wavelength of 1550nm.

    The attached CSV file contains the corresponding intensity profile. 
    The scale is 0.2161um per pixel.
    '''

    # retrieve waveguide intensity and make cutout
    stem = '/Users/bandari/Documents/git.repos/glint_misc/notebooks/data/'
    open_file = open(stem + 'waveguide_intensity.pkl', "rb")
    df_intensity, xycen = pickle.load(open_file)
    open_file.close()
    # cutouts
    buffer = 100 # pix
    waveguide_cutout = df_intensity[int(xycen[1]-buffer):int(xycen[1]+buffer),int(xycen[0]-buffer):int(xycen[0]+buffer)]

    ## USER INPUTS
    input_field = field_complex_A_circ
    mode_field = waveguide_cutout # this variable is only real, but physically the LT0 mode has a phase of zero anyway
    ## END USER INPUT

    overlap_int_complex = np.sum(input_field*mode_field) / np.sqrt( np.sum(np.abs(mode_field)**2) * np.sum(np.abs(input_field)**2) )

    overlap_int = np.abs(overlap_int_complex)**2

    print('overlap_int:',overlap_int)

    if waveguide_cutout.shape != I_PSF_circ.shape:
        print('WAVEGUIDE AND CIRC PSF SHAPES NOT THE SAME!!!')
    if waveguide_cutout.shape != I_PSF_hex.shape:
        print('WAVEGUIDE AND HEX PSF SHAPES NOT THE SAME!!!')


    # plotting

    # radius of first dark ring in um
    wavel = 1.55/1.5255 # um # 1.55 um in air, 1.55um/n = 1.55um/1.5255 = 1.016 um in substrate of index of refraction n
    foc_length = 284.664 # 
    D = 66 # um

    circ_r_um = 1.22 * wavel * foc_length/D
    circ_r_pix = um2pix(circ_r_um)

    print('radius of first dark Airy ring (um):',circ_r_um)

    # define circle for scale in plots
    circ_cen_x = 0
    circ_cen_y = 0 
    circ1 = Circle((circ_cen_x,circ_cen_y),radius=circ_r_pix,color='white',fill=False)
    circ2 = Circle((circ_cen_x,circ_cen_y),radius=circ_r_pix,color='white',fill=False)
    circ3 = Circle((circ_cen_x,circ_cen_y),radius=circ_r_pix,color='white',fill=False)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,10), layout='constrained')

    ax[0,0].set_title('Circular subap')
    ax[0,0].imshow(circ_aperture_simple, cmap='Greys_r', interpolation='nearest')
    ax[0,0].set_xlabel('(arbitrary)')
    ax[0,0].set_ylabel('(arbitrary)')

    ax[0,1].set_title('Hexagonal subap')
    ax[0,1].imshow(hex_aperture_simple, cmap='Greys_r', interpolation='nearest')
    ax[0,1].set_xlabel('(arbitrary)')

    # show normalized profiles of everything
    ax[0,2].plot(np.divide(I_PSF_circ[int(buffer),:],np.max(I_PSF_circ[int(buffer),:])),color='red',label='circle')
    ax[0,2].plot(np.divide(I_PSF_hex[int(buffer),:],np.max(I_PSF_hex[int(buffer),:])),color='blue',label='hexagon')
    ax[0,2].plot(np.divide(waveguide_cutout[int(buffer),:],np.max(waveguide_cutout[int(buffer),:])),color='green',label='waveguide mode')
    ax[0,2].legend()

    ax[1,0].set_title('I (log)')
    ax[1,0].imshow(I_PSF_circ, extent=[-waveguide_cutout.shape[1]/2., waveguide_cutout.shape[1]/2., -waveguide_cutout.shape[0]/2., waveguide_cutout.shape[0]/2. ], alpha=1, norm='log')
    ax[1,0].set_xlabel('pixel')
    ax[1,0].set_ylabel('pixel')
    ax[1,0].add_patch(circ1)

    #ax[1,1].imshow(waveguide_cutout, extent=[-waveguide_cutout.shape[1]/2., waveguide_cutout.shape[1]/2., -waveguide_cutout.shape[0]/2., waveguide_cutout.shape[0]/2. ], origin='lower')

    ax[1,1].set_title('I (log)')
    ax[1,1].imshow(I_PSF_hex, extent=[-waveguide_cutout.shape[1]/2., waveguide_cutout.shape[1]/2., -waveguide_cutout.shape[0]/2., waveguide_cutout.shape[0]/2. ], alpha=1, norm='log')
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

    #plt.show()
    plt.savefig('junk.png')

    '''
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(aperture_simple_choice)
    axs[0].set_title('Aperture')
    axs[1].imshow(I, norm='log')
    axs[1].set_title('Intensity')
    axs[2].imshow(amp)
    axs[2].set_title('Amp')
    axs[3].imshow(arg)
    axs[3].set_title('Arg')

    plt.show()
    '''

    return


if __name__ == '__main__':

    main()

    # pickle results
    '''
    data_list = [amp, arg, I, aperture]
    file = open('junk.pkl', 'wb')
    pickle.dump(data_list, file)
    file.close()
    '''
