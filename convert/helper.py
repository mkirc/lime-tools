import h5py
import numpy as np
from h5py import string_dtype
from h5py import Datatype
from h5py.h5t import TypeID, STR_NULLTERM


def sampleSpherical(npoints, ndim=3):
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
    xMax, yMax, zMax = 0
    for bb in boundingboxes:
        xMax = np.max(xMax, np.max(np.abs(bb[0][1]), np.abs(bb[0][0])))
        yMax = np.max(yMax, np.max(np.abs(bb[1][1]), np.abs(bb[1][0])))
        zMax = np.max(zMax, np.max(np.abs(bb[2][1]), np.abs(bb[2][0])))

    return np.square(xMax**2 + yMax**2 + zMax**2)
