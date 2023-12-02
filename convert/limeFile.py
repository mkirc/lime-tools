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
        self.gpPerBlock = 512
        self.gridGroup = None
        self.gridColumnsGroup = None
        self.idDataset = None
        self.positionDatasets = []
        self.velocityDatasets = []
        self.magfieldDatasets = []
        self.sinkDataset = None
        self.densityDataset = None
        self.gasTemperatureDataset = None
        self.dustTemperatureDataset = None

    def __enter__(self):
        self.file = h5py.File(*self.args)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.__exit__(exc_type, exc_value, traceback)

    def setupFileAttributes(self, radius=0.0, minscale=0.0):
        self.radius = radius
        self.minscale = minscale
        self.createLimeFileAttrs()
    
    def createLimeFileAttrs(self):
        self.file.attrs.create("RADIUS  ", self.radius, dtype=np.float64)
        self.file.attrs.create("MINSCALE", self.minscale, dtype=np.float64)
        self.file.attrs.create("NSOLITER", 0, dtype=np.int32)

    def setupPrimaryGroups(self):
        self.createGridGroup()
        self.createColumsGroup()

    def createGridGroup(self):
        self.gridGroup = self.file.create_group("GRID")
        self.gridGroup.attrs.create("CLASS", "HDU", dtype=nulltermStringType(4))
        self.gridGroup.attrs.create("COLLPAR1", "H2", dtype=nulltermStringType(3))
        self.gridGroup.attrs.create("EXTNAME", "GRID", dtype=nulltermStringType(5))
        self.gridGroup.attrs.create("HDUNUM", 0, dtype=np.int32)

    def createColumsGroup(self):
        self.gridColumnsGroup = self.file.create_group("GRID/columns")
        self.gridColumnsGroup.attrs.create(
            "CLASS", "DATA_GROUP", dtype=nulltermStringType(11)
        )

    def setupPoints(self, nBlocks, nSinks=0, gridpoints=True):
    """needs to be called before property setups. Number of Blocks and sinks need to be known"""
        self.gpPerBlock = 512 if gridpoints else 1
        self.nBlocks = nBlocks
        self.nSinks = nSinks
        self.createIdDataset()
        self.createPositionDatasets()
        self.createSinkDataset()


    def setupDensity(self):
        self.setupPropertyDataset(self.createDensityDataset)

    def setupGasTemperature(self):
        self.setupPropertyDataset(self.createGasTemperatureDataset)

    def setupDustTemperature(self):
        self.setupPropertyDataset(self.createDustTemperatureDataset)

    def setupVelocity(self):
        self.setupPropertyDataset(self.createVelocityDatasets)

    def setupMagfield(self):
        self.setupPropertyDataset(self.createMagfieldDatasets)

    def setupPropertyDataset(self, createFunction):
        """assumes setupPrimaryGroups called before"""
        if not self.gridColumnsGroup:
            raise AttributeError(
                "No GRID/columns Group, run setupPointsAndSinks first."
            )
        return createFunction()

    def createIdDataset(self):
        self.idDataset = self.file.create_dataset(
            "GRID/columns/ID",
            (self.nBlocks * self.gpPerBlock + self.nSinks),
            dtype=np.uint32,
            data=np.arange(self.nBlocks * self.gpPerBlock + self.nSinks),
        )
        self.idDataset.attrs.create("CLASS", "COLUMN", dtype=nulltermStringType(7))
        self.idDataset.attrs.create("COL_NAME", "ID", dtype=nulltermStringType(3))
        self.idDataset.attrs.create("UNIT", "", dtype=nulltermStringType(1))

    def createSinkDataset(self):
        self.sinkDataset = self.file.create_dataset(
            "GRID/columns/IS_SINK",
            (self.nBlocks * self.gpPerBlock + self.nSinks),
            dtype=np.int16,
        )
        self.idDataset.attrs.create("CLASS", "COLUMN", dtype=nulltermStringType(7))
        self.idDataset.attrs.create("COL_NAME", "IS_SINK", dtype=nulltermStringType(8))
        self.idDataset.attrs.create("UNIT", "", dtype=nulltermStringType(1))

    def createPositionDatasets(self):
        self.create3DDatasetForNameAndUnit(self.positionDatasets, "X", "m")

    def createVelocityDatasets(self):
        self.create3DDatasetForNameAndUnit(self.velocityDatasets, "VEL", "m/s")

    def createMagfieldDatasets(self):
        self.create3DDatasetForNameAndUnit(self.magfieldDatasets, "B_FIELD", "T")

    def create3DDatasetForNameAndUnit(self, dataset, name, unitName):
        """creates dataset of shape (len(blocks)* points/block + len(sinkpoints),)"""
        for i in range(1, 4):
            dataset.append(
                self.file.create_dataset(
                    f"GRID/columns/{name}{i}",
                    (self.nBlocks * self.gpPerBlock + self.nSinks),
                    dtype=np.float64,
                )
            )
            dataset[i - 1].attrs.create("CLASS", "COLUMN", dtype=nulltermStringType(7))
            dataset[i - 1].attrs.create(
                "COL_NAME", f"{name}{i}", dtype=nulltermStringType(len(name) + 2)
            )
            dataset[i - 1].attrs.create(
                "UNIT", f"{unitName}", dtype=nulltermStringType(len(unitName) + 1)
            )

    def createDensityDataset(self):
        self.densityDataset = self.file.create_dataset(
            "GRID/columns/DENSITY1",
            (self.nBlocks * self.gpPerBlock + self.nSinks),
            dtype=np.float32,
        )
        self.densityDataset.attrs.create("CLASS", "COLUMN", dtype=nulltermStringType(7))
        self.densityDataset.attrs.create(
            "COL_NAME", "DENSITY1", dtype=nulltermStringType(9)
        )
        self.densityDataset.attrs.create("UNIT", "kg/m^3", dtype=nulltermStringType(7))

    def createGasTemperatureDataset(self):
        totalNumber = self.nBlocks * self.gpPerBlock + self.nSinks
        self.gasTemperatureDataset = self.file.create_dataset(
            "GRID/columns/TEMPKNTC",
            totalNumber,
            dtype=np.float32,
            data=np.array([2.7548] * totalNumber),
        )
        self.gasTemperatureDataset.attrs.create(
            "CLASS", "COLUMN", dtype=nulltermStringType(7)
        )
        self.gasTemperatureDataset.attrs.create(
            "COL_NAME", "TEMPKNTC", dtype=nulltermStringType(9)
        )
        self.gasTemperatureDataset.attrs.create(
            "UNIT", "K", dtype=nulltermStringType(2)
        )

    def createDustTemperatureDataset(self):
        self.dustTemperatureDataset = self.file.create_dataset(
            "GRID/columns/TEMPDUST",
            (self.nBlocks * self.gpPerBlock + self.nSinks),
            dtype=np.float32,
        )
        self.dustTemperatureDataset.attrs.create(
            "CLASS", "COLUMN", dtype=nulltermStringType(7)
        )
        self.dustTemperatureDataset.attrs.create(
            "COL_NAME", "TEMPDUST", dtype=nulltermStringType(9)
        )
        self.dustTemperatureDataset.attrs.create(
            "UNIT", "K", dtype=nulltermStringType(2)
        )

    def writeBlocks(self, blocks):
        allGridpoints = self.nBlocks * self.gpPerBlock

        iBlock = 0
        for block in blocks:
            # write position data
            self.writeGridpointPositions(block, iBlock)

            # write property data
            self.writeDensities(block, iBlock)
            self.writeGasTemperatures(block, iBlock)
            self.writeDustTemperatures(block, iBlock)
            self.writeVelocities(block, iBlock)
            self.writeMagfield(block, iBlock)

            iBlock += 1

    def writeSinks(self, sinkpoints):
        xSink, ySink, zSink = sinkpoints
        allGridpoints = self.nBlocks * self.gpPerBlock

        # write sinkpoint positions
        self.positionDatasets[0][allGridpoints : allGridpoints + self.nSinks] = xSink
        self.positionDatasets[1][allGridpoints : allGridpoints + self.nSinks] = ySink
        self.positionDatasets[2][allGridpoints : allGridpoints + self.nSinks] = zSink

        # write sinkpoint bitmask
        self.sinkDataset[0:allGridpoints] = np.zeros(allGridpoints)
        self.sinkDataset[allGridpoints:] = np.ones(self.nSinks)

    def writeGridpointPositions(self, block, iBlock):
        self.positionDatasets[0][
            iBlock * self.gpPerBlock : (iBlock + 1) * self.gpPerBlock
        ] = block.gridpoints[:, 0]
        self.positionDatasets[1][
            iBlock * self.gpPerBlock : (iBlock + 1) * self.gpPerBlock
        ] = block.gridpoints[:, 1]
        self.positionDatasets[2][
            iBlock * self.gpPerBlock : (iBlock + 1) * self.gpPerBlock
        ] = block.gridpoints[:, 2]

    def writeDensities(self, block, iBlock):
        if self.densityDataset is not None and block.densities is not None:
            self.densityDataset[
                iBlock * self.gpPerBlock : (iBlock + 1) * self.gpPerBlock
            ] = block.densities[:]

    def writeGasTemperatures(self, block, iBlock):
        if self.gasTemperatureDataset is not None and block.temperatures is not None:
            self.gasTemperatureDataset[
                iBlock * self.gpPerBlock : (iBlock + 1) * self.gpPerBlock
            ] = block.temperatures[:]

    def writeDustTemperatures(self, block, iBlock):
        if (
            self.dustTemperatureDataset is not None
            and block.dusttemperatures is not None
        ):
            self.dustTemperatureDataset[
                iBlock * self.gpPerBlock : (iBlock + 1) * self.gpPerBlock
            ] = block.dusttemperatures[:]

    def writeVelocities(self, block, iBlock):
        if len(self.velocityDatasets) > 0 and block.velocities is not None:
            self.velocityDatasets[0][
                iBlock * self.gpPerBlock : (iBlock + 1) * self.gpPerBlock
            ] = block.velocities[:, 0]
            self.velocityDatasets[1][
                iBlock * self.gpPerBlock : (iBlock + 1) * self.gpPerBlock
            ] = block.velocities[:, 1]
            self.velocityDatasets[2][
                iBlock * self.gpPerBlock : (iBlock + 1) * self.gpPerBlock
            ] = block.velocities[:, 2]

    def writeMagfield(self, block, iBlock):
        if len(self.magfieldDatasets) > 0 and block.magfluxes is not None:
            self.magfieldDatasets[0][
                iBlock * self.gpPerBlock : (iBlock + 1) * self.gpPerBlock
            ] = block.magfluxes[:, 0]
            self.magfieldDatasets[1][
                iBlock * self.gpPerBlock : (iBlock + 1) * self.gpPerBlock
            ] = block.magfluxes[:, 1]
            self.magfieldDatasets[2][
                iBlock * self.gpPerBlock : (iBlock + 1) * self.gpPerBlock
            ] = block.magfluxes[:, 2]
