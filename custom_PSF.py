import numpy as np
import matplotlib.pyplot as plt


#Create Fizeau PSF

phase=np.pi/2.0*0.0
D=8250 #mm
padfactor=16.0
deltaspacing = 100 #mm
centerx=D*padfactor/deltaspacing/2.0 #pixels
centery=D*padfactor/deltaspacing/2.0 #pixels
hwstamp = D/2.0/deltaspacing #pixels

def LBT_field(phase):
    # returns the physical aperture, and the complex aperture*phase

    lenx = D*padfactor/deltaspacing #pixels
    leny = D*padfactor/deltaspacing #pixels

    xpos=np.arange(-D/2.0*padfactor,D/2.0*padfactor,deltaspacing)
    ypos=np.arange(-D/2.0*padfactor,D/2.0*padfactor,deltaspacing)
    xx,yy =np.meshgrid(xpos,ypos)
    xsx = xx - 7208.5
    xdx = xx + 7208.5
    rr=np.sqrt(xx**2+yy**2)
    rrsx= np.sqrt(xsx**2+yy**2)
    rrdx= np.sqrt(xdx**2+yy**2)
    sxAperture=rrsx<D/2.0
    sxAperturephase =sxAperture * 1.0
    dxAperture=rrdx<D/2.0
    dxAperturephase =dxAperture * np.exp(1j*phase)
    LBTAperture = sxAperture + dxAperture
    LBTAperturephase = sxAperturephase + dxAperturephase

    return LBTAperture, LBTAperturephase


def circ_field(phase):
    # returns the physical aperture, and the complex aperture*phase

    lenx = D*padfactor/deltaspacing #pixels
    leny = D*padfactor/deltaspacing #pixels

    xpos=np.arange(-D/2.0*padfactor,D/2.0*padfactor,deltaspacing)
    ypos=np.arange(-D/2.0*padfactor,D/2.0*padfactor,deltaspacing)
    xx,yy =np.meshgrid(xpos,ypos)
    xdx = xx + 7208.5
    rr=np.sqrt(xx**2+yy**2)
    rrdx= np.sqrt(xdx**2+yy**2)
    dxAperture=rrdx<D/2.0
    dxAperturephase =dxAperture * np.exp(1j*phase)
    aperture = dxAperture
    aperturephase = dxAperturephase


    return aperture, aperturephase


LBTAperture, LBTAperturephase = LBT_field(phase)
circ_aperture, circ_aperture_phase = circ_field(phase)

aperture_phase_choice = circ_aperture_phase
aperture_simple_choice = circ_aperture

#Create LBT PSF
FTAp = np.fft.fft2(aperture_phase_choice)
FTAp=np.fft.fftshift(FTAp)
FTAp=FTAp[int(centerx-hwstamp):int(centerx+hwstamp),int(centery-hwstamp):int(centery+hwstamp)]


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
axs[1].imshow(I)
axs[1].set_title('Intensity')
axs[2].imshow(AmpPE)
axs[2].set_title('Amp')
axs[3].imshow(ArgPE)
axs[3].set_title('Arg')

plt.show()