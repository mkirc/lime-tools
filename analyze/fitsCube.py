import pathlib
import aplpy
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import fits  # We use fits to open the actual data file
from matplotlib.colors import LogNorm
from astropy.wcs import WCS

from spectral_cube import SpectralCube

outDir = pathlib.Path(__file__).parent.absolute() / 'out'
inDir = pathlib.Path(__file__).parent.parent.parent.absolute() / 'wip' / 'models' / 'simACR'
# inDir = pathlib.Path(__file__).parent.absolute() / 'in'

def plotJanskyPerPixel():

    u.add_enabled_units(u.def_unit(['JY/PIXEL'], represents=u.Jy))

    file = fits.open(str(inDir.joinpath("image0_Jansky-per-px.fits")))

    file[0].header["CUNIT3"] = "m/s"

    # cube = SpectralCube.read(file)

    image_hdu = file[0]
    try:
        image_wcs = WCS(image_hdu)
    except Exception:
        image_wcs = None

    plotSpectralCube(image_hdu, image_wcs, 'jyPp.pdf')
    file.close()

def plotSimpleFlashJyPP():

    inDir = pathlib.Path(__file__).parent.absolute() / 'in'
    file = fits.open(
        str(inDir.joinpath(
                'image_low_res_rfl9_cnt_0000_1362um_incl_0_phi_0_240000AU_64'
                'pixel_jyPixel.fits')
        ))

    plotFlashCube(file[0])

def plotOpticalDepth():

    file = fits.open(str(inDir.joinpath("image0_Tau.fits")))

    file[0].header["CUNIT3"] = "m/s"

    # cube = SpectralCube.read(file)

    image_hdu = file[0]
    image_wcs = WCS(image_hdu)

    plotSpectralCube(image_hdu, image_wcs, 'tau.pdf')
    file.close()

def plotKelvin():

    file = fits.open(str(inDir.joinpath("image0_Kelvin.fits")))

    file[0].header["CUNIT3"] = "m/s"

    # cube = SpectralCube.read(file)

    image_hdu = file[0]
    image_wcs = WCS(image_hdu)

    plotSpectralCube(image_hdu, image_wcs, 'kelvin.pdf')
    file.close()

def plotFlashCube(imageHDU, name='JyPP-flash.pdf'):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    im = ax.imshow(imageHDU.data[0,:,:])
    # Add a colorbar
    cbar = plt.colorbar(im, pad=0.07)

    # Add axes labels
    ax.set_xlabel("Right Ascension", fontsize=14)
    ax.set_ylabel("Declination", fontsize=14)

    plt.savefig(str(outDir.joinpath(name)), bbox_inches="tight")

def plotSpectralCube(imageHDU, projection, name="spec-quick.png"):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=projection, slices=("x", "y", 30, 1))
    im = ax.imshow(imageHDU.data[0, 30, :, :])

    # Add a colorbar
    cbar = plt.colorbar(im, pad=0.07)
    cbar.set_label(imageHDU.header["BUNIT"], size=14)

    # Add axes labels
    ax.set_xlabel("Right Ascension", fontsize=14)
    ax.set_ylabel("Declination", fontsize=14)

    # plt.plot(cube[:,:50,50])
    plt.savefig(str(outDir.joinpath(name)), bbox_inches="tight")

if __name__ == "__main__":

    # plotJanskyPerPixel()
    plotSimpleFlashJyPP()
    # plotKelvin()
    # plotOpticalDepth()

# Extract a single spectrum through the data cube
# cube[:,50,50].quicklook('spec-quick.png')


# # zero'th moment
# m0 = cube.moment(order=0)
# print(m0.unit)
# m0.quicklook("out/m0.png")

# # first moment
# m1 = cube.moment(order=1)
# print(m1.unit)
# m1.quicklook("out/m1.png")


# f  = aplpy.FITSFigure(m0.hdu)
# f.show_colorscale()
# f.save('m0.png')

# print(cube)
