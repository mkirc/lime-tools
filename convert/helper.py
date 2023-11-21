import h5py
import numpy as np
from h5py import string_dtype
from h5py import Datatype
from h5py.h5t import TypeID, STR_NULLTERM

def sampleSphere(npoints):
    phi   = np.random.uniform(0, 2 * np.pi, npoints)
    theta = np.arccos(np.random.uniform(-1, 1, npoints))
    u     = np.random.uniform(0, 1, npoints)

    coords = np.zeros((npoints,3))

    for i in range(npoints):
        r = u[i] ** 1/3
        coords[i, 0] = r * np.sin(theta[i]) * np.cos(phi[i])
        coords[i, 1] = r * np.sin(theta[i]) * np.sin(phi[i])
        coords[i, 2] = r * np.cos(theta[i])

    return coords

def sampleSphereSurface(npoints, ndim=3):
    """generates points randomly placed on unit sphere"""
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def centerAxis(arraylike):
    return arraylike - ((np.max(arraylike) + np.min(arraylike)) / 2)


def flatten3DValues(valuesX, valuesY, valuesZ):
    """takes 3d list of np.arrays of shape (x,y,z) returns 3d np.array of row-major
    flattened np.arrays of shape (x*y*z,)"""

    return np.array([valuesX.flatten(), valuesY.flatten(), valuesZ.flatten()]).T


def radiusForBoundingboxes(boundingboxes):
    """returns 3D radius of sphere guaranteed to envelop region
    defined in list of boundingboxes"""
    xMax = yMax = zMax = 0
    for bb in boundingboxes:
        xMax = max(xMax, max(np.abs(bb[0][1]),abs(bb[0][0])))
        yMax = max(yMax, max(abs(bb[1][1]), abs(bb[1][0])))
        zMax = max(zMax, max(abs(bb[2][1]), abs(bb[2][0])))

    return np.sqrt(xMax**2 + yMax**2 + zMax**2)
