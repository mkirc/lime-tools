import aplpy
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import fits  # We use fits to open the actual data file
from matplotlib.colors import LogNorm
from astropy.wcs import WCS

from spectral_cube import SpectralCube

# u.add_enabled_units(u.def_unit(['JY/PIXEL'], represents=u.<correct_unit>))

file = fits.open('../../wip/models/advanced/image0_Kelvin.fits')

file[0].header["CUNIT3"] = 'm/s'


cube = SpectralCube.read(file)

image_hdu = file[0]
image_wcs = WCS(image_hdu)


# Slice the cube along the spectral axis, and display a quick image
cube[30,:,:].quicklook('xy-quick.png')


print(image_wcs)

fig = plt.figure()
ax = fig.add_subplot(111, projection=image_wcs, slices=('x','y',30,1))
im = ax.imshow(image_hdu.data[0,30,:,:])

# Add a colorbar
cbar = plt.colorbar(im, pad=.07)
cbar.set_label(image_hdu.header["BUNIT"], size = 16)

# Add axes labels
ax.set_xlabel("Right Ascension", fontsize = 16)
ax.set_ylabel("Declination", fontsize = 16)

# plt.plot(cube[:,:50,50])
plt.savefig('spec-quick.png', bbox_inches="tight")

# Extract a single spectrum through the data cube
# cube[:,50,50].quicklook('spec-quick.png')


# zero'th moment
m0 = cube.moment(order=0)
print(m0.unit)
m0.quicklook('m0.png')
m1 = cube.moment(order=1)
print(m1.unit)
m1.quicklook('m1.png')


# f  = aplpy.FITSFigure(m0.hdu)
# f.show_colorscale()
# f.save('m0.png')

print(cube)
file.close()
