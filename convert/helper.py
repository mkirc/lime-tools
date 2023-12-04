import h5py
import argparse
import textwrap
import numpy as np
from h5py import string_dtype
from h5py import Datatype
from h5py.h5t import TypeID, STR_NULLTERM


def createArgumentParser():
    arg_parser = argparse.ArgumentParser(
        description="Converts FLASH checkpoint files to valid LIME input.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
        TODO
        """
        ),
    )
    arg_parser.add_argument(
        "inFile",
        metavar="FLASH file path",
        type=str,
        help="Path of FLASH checkpoint file",
    )
    arg_parser.add_argument(
        "outFile", metavar="LIME file path", type=str, help="Path of LIME file"
    )
    arg_parser.add_argument(
        "-b",
        "--blocks",
        type=int,
        help="Number of FLASH leaf blocks to include. Defaults to all Blocks found in FLASH file.",
    )
    arg_parser.add_argument(
        "-s",
        "--sinks",
        type=int,
        help="Number of LIME sink points to include. Defaults to 1000",
    )
    arg_parser.add_argument(
        "-r",
        "--radscale",
        type=float,
        help="Scale factor to apply to radius of sink points. Defaults to 1",
    )

    return arg_parser


def sampleSphere(npoints):
    """generates points randomly placed in volume of unit sphere"""
    phi = np.random.uniform(0, 2 * np.pi, npoints)
    theta = np.arccos(np.random.uniform(-1, 1, npoints))
    u = np.random.uniform(0, 1, npoints)

    coords = np.zeros((npoints, 3))

    for i in range(npoints):
        r = u[i] ** 1 / 3
        coords[i, 0] = r * np.sin(theta[i]) * np.cos(phi[i])
        coords[i, 1] = r * np.sin(theta[i]) * np.sin(phi[i])
        coords[i, 2] = r * np.cos(theta[i])

    return coords


def sampleSphereSurface(npoints, ndim=3):
    """generates points randomly placed on surface of unit sphere"""
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def centerAxis(arraylike):
    return arraylike - ((np.max(arraylike) + np.min(arraylike)) / 2)


def flatten3DValues(valuesX, valuesY, valuesZ):
    """takes three-element-list of np.arrays of shape (nx,ny,nz),
    column-major ("Fortran-style") flattening them and
    returns np.array shape (nx*ny*nz,3)"""

    return np.array(
        [
            valuesX.flatten(order="F"),
            valuesY.flatten(order="F"),
            valuesZ.flatten(order="F"),
        ]
    ).T


def radiusForBoundingboxes(boundingboxes):
    """returns 3D radius of sphere guaranteed to envelop region
    defined in list of boundingboxes"""
    xMax = yMax = zMax = 0
    for bb in boundingboxes:
        xMax = max(xMax, max(np.abs(bb[0][1]), abs(bb[0][0])))
        yMax = max(yMax, max(abs(bb[1][1]), abs(bb[1][0])))
        zMax = max(zMax, max(abs(bb[2][1]), abs(bb[2][0])))

    return np.sqrt(xMax**2 + yMax**2 + zMax**2)
