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


def nulltermStringType(length):
    type_id = TypeID.copy(h5py.h5t.C_S1)
    type_id.set_size(length)
    type_id.set_strpad(STR_NULLTERM)

    return h5py.Datatype(type_id)


def setupLIMEStage1(outFile, n_blocks, radius=0.0, minscale=0.0):
    outFile.attrs.create("RADIUS  ", radius, dtype=np.float64)
    outFile.attrs.create("MINSCALE", minscale, dtype=np.float64)
    outFile.attrs.create("NSOLITER", 0, dtype=np.int32)
    grid_grp = outFile.create_group("GRID")
    grid_grp.attrs.create("CLASS", "HDU", dtype=nulltermStringType(4))
    grid_grp.attrs.create("COLLPAR1", "H2", dtype=nulltermStringType(3))
    grid_grp.attrs.create("EXTNAME", "GRID", dtype=nulltermStringType(5))
    grid_grp.attrs.create("HDUNUM", 0, dtype=np.int32)
    cols_grp = outFile.create_group("GRID/columns")
    cols_grp.attrs.create("CLASS", "DATA_GROUP", dtype=nulltermStringType(11))
    lime_id = outFile.create_dataset(
        "GRID/columns/ID",
        (n_blocks * 512 + 296),
        dtype=np.uint32,
        data=np.arange(n_blocks * 512 + 296),
    )
    lime_id.attrs.create("CLASS", "COLUMN", dtype=nulltermStringType(7))
    lime_id.attrs.create("COL_NAME", "ID", dtype=nulltermStringType(3))
    lime_id.attrs.create("POSITION", 1, dtype=np.int32)
    lime_id.attrs.create("UNIT", "", dtype=nulltermStringType(1))
    lime_is_sink = outFile.create_dataset(
        "GRID/columns/IS_SINK", (n_blocks * 512 + 296), dtype=np.int16
    )
    lime_is_sink.attrs.create("CLASS", "COLUMN", dtype=nulltermStringType(7))
    lime_is_sink.attrs.create("COL_NAME", "IS_SINK", dtype=nulltermStringType(8))
    lime_is_sink.attrs.create("POSITION", 5, dtype=np.int32)
    lime_is_sink.attrs.create("UNIT", "", dtype=nulltermStringType(1))

    lime_x1 = outFile.create_dataset(
        "GRID/columns/X1", (n_blocks * 512 + 296), dtype=np.float64
    )
    lime_x1.attrs.create("CLASS", "COLUMN", dtype=nulltermStringType(7))
    lime_x1.attrs.create("COL_NAME", "X1", dtype=nulltermStringType(3))
    lime_x1.attrs.create("POSITION", 2, dtype=np.int32)
    lime_x1.attrs.create("UNIT", "m", dtype=nulltermStringType(2))

    lime_x2 = outFile.create_dataset(
        "GRID/columns/X2", (n_blocks * 512 + 296), dtype=np.float64
    )
    lime_x2.attrs.create("CLASS", "COLUMN", dtype=nulltermStringType(7))
    lime_x2.attrs.create("COL_NAME", "X2", dtype=nulltermStringType(3))
    lime_x2.attrs.create("POSITION", 3, dtype=np.int32)
    lime_x2.attrs.create("UNIT", "m", dtype=nulltermStringType(2))

    lime_x3 = outFile.create_dataset(
        "GRID/columns/X3", (n_blocks * 512 + 296), dtype=np.float64
    )
    lime_x3.attrs.create("CLASS", "COLUMN", dtype=nulltermStringType(7))
    lime_x3.attrs.create("COL_NAME", "X3", dtype=nulltermStringType(3))
    lime_x3.attrs.create("POSITION", 4, dtype=np.int32)
    lime_x3.attrs.create("UNIT", "m", dtype=nulltermStringType(2))

    return lime_x1, lime_x2, lime_x3, lime_is_sink
