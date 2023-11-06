import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import astropy.units as u
from astropy.utils.data import download_file
from astropy.io import fits  # We use fits to open the actual data file

from astropy.utils import data
data.conf.remote_timeout = 60

from spectral_cube import SpectralCube

# from astroquery.esasky import ESASky
# from astroquery.utils import TableList
from astropy.wcs import WCS

# from reproject import reproject_interp


file = fits.open('../../wip/models/advanced/image0.fits')

file[0].header["CUNIT3"] = 'm/s'

cube = SpectralCube.read(file)

file.close()

plt.plot(cube[:,50,50])
plt.show()

m0 = cube.moment(order=0)

import aplpy

f  = aplpy.FITSFigure(m0.hdu)
f.show_colorscale()
f.save('m0.png')

print(cube)
