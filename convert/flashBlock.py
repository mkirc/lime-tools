import sys
import h5py

import numpy as np

from helper import flatten3DValues


class FlashFactory:
    def __init__(self, flash_file):
        self.file = flash_file
        self.bb = self.file["bounding box"]  # needs no get(), bb is required
        self.temperatures = self.file.get("temp")
        self.dusttemperatures = self.file.get("tdus")
        self.refinementLevels = self.file.get("refine level")
        self.blockSizes = self.file.get("block size")
        self.radius = np.sqrt(
            np.max(self.blockSizes[0][0]) ** 2
            + np.max(self.blockSizes[0][1]) ** 2
            + np.max(self.blockSizes[0][2]) ** 2
        )
        self.minscale = (
            np.max(self.blockSizes[0]) / (2 ** (np.max(self.refinementLevels) - 1)) / 8
        )
        self.vels = (
            self.file.get("velx"),
            self.file.get("vely"),
            self.file.get("velz"),
        )
        self.mags = (
            self.file.get("magx"),
            self.file.get("magy"),
            self.file.get("magz"),
        )
        self.densities = self.file.get("dens")
        self.blocks = np.array(self.file["node type"])
        self.leaves = np.where(self.blocks == 1)[0]  # node type == 1 -> leaf
        self.gpIndices = np.meshgrid(
            *[range(nib) for nib in self.densities[0].shape], indexing="ij"
        )

    def generateBlocksForSlice(self, blockslice):
        for blockId in self.leaves[blockslice]:
            yield self.createBlock(blockId)

    def createBlock(self, blockId):
        return FlashBlock(
            blockId,
            self.gpIndices,
            self.bb[blockId],
            self.temperaturesForBlock(blockId),
            self.dusttemperaturesForBlock(blockId),
            self.densitiesForBlock(blockId),
            self.velocitiesForBlock(blockId),
            self.magfluxesForBlock(blockId),
        )

    def temperaturesForBlock(self, blockId):
        try:
            return self.temperatures[blockId]
        except TypeError:
            return None

    def dusttemperaturesForBlock(self, blockId):
        try:
            return self.dusttemperatures[blockId]
        except TypeError:
            return None

    def densitiesForBlock(self, blockId):
        try:
            return self.densities[blockId]
        except TypeError:
            return None

    def velocitiesForBlock(self, blockId):
        if not self.vels[0]:
            # raise AttributeError('No velocities present in flashfile.')
            return None
        return (self.vels[0][blockId], self.vels[1][blockId], self.vels[2][blockId])

    def magfluxesForBlock(self, blockId):
        if not self.mags[0]:
            # raise AttributeError('No mag field present in flashfile.')
            return None
        return (self.mags[0][blockId], self.mags[1][blockId], self.mags[2][blockId])


class FlashBlock:
    def __init__(self, blockId, gpIndices, bb, temp, tempdust, dens, vels, mags):
        self.id = blockId
        self._Ix, self._Iy, self._Iz = gpIndices

        self.gridpoints = self.gridpointsForBoundingbox(bb)
        self.temperatures = self.temperatures(temp)
        self.dusttemperatures = self.dusttemperatures(tempdust)
        self.densities = self.densities(dens)
        self.velocities = self.velocities(vels)
        self.magfluxes = self.magfluxes(mags)
        self.hullpoints = self.hullpoints()

    def gridpointsForBoundingbox(self, bb):
        """takes np.array of shape (3,2) [x,y,z, (upper/lower)]. returns
        row-major flattened np.array of coordinates of shape (self.nxb *
        self.nyb * self.nzb, 3)"""

        (dx, x0), (dy, y0), (dz, z0) = [
            (bb[i][1] - bb[i][0], bb[i][0]) for i in range(3)
        ]
        return (
            np.array([self._Ix * dx + x0, self._Iy * dy + y0, self._Iz * dz + z0])
            .reshape(3, -1)
            .T
        )

    def hullpoints(self):
        """hullpoint has either a index of 0 or n[xyz]b. returns bitmask for gridpoints,
        1 indicating hullpoint, 0 indicating inner point. This whole method is pretty pointless
        for anything other than looking at one block"""
        out = []
        dirty_nxb = np.max(self._Ix)  # TODO: this is not generic, refactor.
        gridpointIndices = np.array([self._Ix, self._Iy, self._Iz]).reshape(3, -1).T

        for p in gridpointIndices:
            if 0 in p or dirty_nxb in p:
                out.append(1)
            else:
                out.append(0)

        return out

    def temperatures(self, temperatures):
        if temperatures is not None:
            return temperatures.flatten()

    def dusttemperatures(self, dusttemperatures):
        if dusttemperatures is not None:
            return dusttemperatures.flatten()

    def densities(self, densities):
        if densities is not None:
            return densities.flatten()

    def velocities(self, velocities):
        if velocities is not None:
            return flatten3DValues(velocities[0], velocities[1], velocities[2])

    def magfluxes(self, magfluxes):
        if magfluxes is not None:
            return flatten3DValues(magfluxes[0], magfluxes[1], magfluxes[2])
