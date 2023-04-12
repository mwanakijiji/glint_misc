import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2


#Create Fizeau PSF

phase=np.pi/2.0*0.0
D=8250 #mm
padfactor=16.0
deltaspacing = 100 #mm
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


LBTAperture, LBTAperturephase = LBT_field(phase)
circ_aperture, circ_aperture_phase = circ_field(phase)
hex_aperture, hex_aperture_phase = hex_field(phase)

aperture_phase_choice = hex_aperture_phase
aperture_simple_choice = hex_aperture

#Create LBT PSF
FTAp = np.fft.fft2(aperture_phase_choice)
FTAp = np.fft.fftshift(FTAp)
FTAp = FTAp[int(centerx-hwstamp):int(centerx+hwstamp),int(centery-hwstamp):int(centery+hwstamp)]


I= np.real(FTAp * np.conj(FTAp))

#Now extract results from the PSF  
cutI=I[0:len(I),0:len(I)-4]   #getting the software aperture offset by a certain amount

padI=np.pad(cutI,len(cutI),'edge')

#This was needed to get rid of the checkerboard pattern
padI = np.fft.fftshift(padI)

PhaseExtract=np.fft.fft2(padI)
PhaseExtract = np.fft.fftshift(PhaseExtract)

AmpPE = np.absolute(PhaseExtract)
ArgPE = np.angle(PhaseExtract)

CenterPix = len(PhaseExtract)/2.0
IntPix = 150.0;

CenterPhase = ArgPE[int(CenterPix),int(CenterPix)]
InterfPhase = ArgPE[int(CenterPix),int(IntPix)] * 180.0/np.pi

print("Phase of Interference fringes = ",   InterfPhase)

fig, axs = plt.subplots(1, 4)
axs[0].imshow(aperture_simple_choice)
axs[0].set_title('Aperture')
axs[1].imshow(I, norm='log')
axs[1].set_title('Intensity')
axs[2].imshow(AmpPE)
axs[2].set_title('Amp')
axs[3].imshow(ArgPE)
axs[3].set_title('Arg')

plt.show()