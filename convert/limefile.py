import h5py

import numpy as np

from h5py import string_dtype
from h5py import Datatype
from h5py.h5t import TypeID, STR_NULLTERM


def nulltermStringType(length):
    type_id = TypeID.copy(h5py.h5t.C_S1)
    type_id.set_size(length)
    type_id.set_strpad(STR_NULLTERM)

    return h5py.Datatype(type_id)

class LimeFile:

    def __init__(self, *args):
        self.args = args
        self.nBlocks = 0
        self.nSinks = 0
        self.radius = 0.0
        self.minscale = 0.0
        self.positionDatasets = []

    def __enter__(self):

        self.file = h5py.File(*self.args)

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        self.file.__exit__(exc_type, exc_value, traceback)

    def setupStageOne(self, nBlocks, nSinks, radius=0.0, minscale=0.0, gridpoints=True):
        self.pointsPerBlock = 512 if gridpoints else 1
        self.nBlocks = nBlocks
        self.nSinks = nSinks
        self.radius = radius
        self.minscale = minscale
        self.createLimeFileAttrs()
        self.createGridGroup()
        self.createColumsGroup()
        self.createIdDataset()
        self.createPositionDatasets()

        return self.positionDatasets

    def createLimeFileAttrs(self):
        self.file.attrs.create("RADIUS  ", self.radius, dtype=np.float64)
        self.file.attrs.create("MINSCALE", self.minscale, dtype=np.float64)
        self.file.attrs.create("NSOLITER", 0, dtype=np.int32)

    def createGridGroup(self):
        self.gridGroup = self.file.create_group("GRID")
        self.gridGroup.attrs.create("CLASS", "HDU", dtype=nulltermStringType(4))
        self.gridGroup.attrs.create("COLLPAR1", "H2", dtype=nulltermStringType(3))
        self.gridGroup.attrs.create("EXTNAME", "GRID", dtype=nulltermStringType(5))
        self.gridGroup.attrs.create("HDUNUM", 0, dtype=np.int32)

    def createColumsGroup(self):
        self.columnsGroup = self.file.create_group("GRID/columns")
        self.columnsGroup.attrs.create("CLASS", "DATA_GROUP", dtype=nulltermStringType(11))

    def createIdDataset(self):
        self.idDataset = self.file.create_dataset(
            "GRID/columns/ID",
            (self.nBlocks * self.pointsPerBlock + self.nSinks),
            dtype=np.uint32,
            data=np.arange(self.nBlocks * self.pointsPerBlock + self.nSinks),
        )
        self.idDataset.attrs.create("CLASS", "COLUMN", dtype=nulltermStringType(7))
        self.idDataset.attrs.create("COL_NAME", "ID", dtype=nulltermStringType(3))
        self.idDataset.attrs.create("POSITION", 1, dtype=np.int32)
        self.idDataset.attrs.create("UNIT", "", dtype=nulltermStringType(1))

    def createSinkDataset(self):
        self.sinkDataset = self.file.create_dataset(
            "GRID/columns/IS_SINK", (self.nBlocks * self.pointsPerBlock + self.nSinks), dtype=np.int16
        )
        self.idDataset.attrs.create("CLASS", "COLUMN", dtype=nulltermStringType(7))
        self.idDataset.attrs.create("COL_NAME", "IS_SINK", dtype=nulltermStringType(8))
        self.idDataset.attrs.create("POSITION", 5, dtype=np.int32)
        self.idDataset.attrs.create("UNIT", "", dtype=nulltermStringType(1))

    def createPositionDatasets(self):
        for i in range(1, 4):
            self.positionDatasets.append(
                self.file.create_dataset(
                    f"GRID/columns/X{i}",
                    (self.nBlocks * self.pointsPerBlock + self.nSinks),
                    dtype=np.float64
                )
            )
            self.positionDatasets[i-1].attrs.create("CLASS", "COLUMN", dtype=nulltermStringType(7))
            self.positionDatasets[i-1].attrs.create("COL_NAME", f"X{i}", dtype=nulltermStringType(3))
            self.positionDatasets[i-1].attrs.create("POSITION", i+1, dtype=np.int32)
            self.positionDatasets[i-1].attrs.create("UNIT", "m", dtype=nulltermStringType(2))
