import sys
import h5py

import numpy as np

from helper import flatten3DValues


class FlashBlockFactory:
    def __init__(self, flash_file):
        self.file = flash_file
        self.bb = self.file["bounding box"]
        self.temperatures = self.file["temp"]
        self.dusttemperatures = self.file["tdus"]
        self.vels = (self.file["velx"], self.file["vely"], self.file["velz"])
        self.mags = (self.file["magx"], self.file["magy"], self.file["magz"])
        self.densities = self.file["dens"]
        self.blocks = np.array(self.file["node type"])
        self.leaves = np.where(self.blocks == 1)[0]  # node type == 1 -> leaf
        self.indices = np.meshgrid(
            *[range(nib) for nib in self.densities[0].shape], indexing="ij"
        )

    def velocitiesForBlock(self, blockId):
        return (self.vels[0][blockId], self.vels[1][blockId], self.vels[2][blockId])

    def magfluxesForBlock(self, blockId):
        return (self.mags[0][blockId], self.mags[1][blockId], self.mags[2][blockId])

    def generateBlocksForSlice(self, blockslice):
        for blockId in self.leaves[blockslice]:
            yield self.createBlock(blockId)

    def createBlock(self, blockId):
        return FlashBlock(
            blockId,
            self.indices,
            self.bb[blockId],
            self.temperatures[blockId],
            self.dusttemperatures[blockId],
            self.densities[blockId],
            self.velocitiesForBlock(blockId),
            self.magfluxesForBlock(blockId),
        )


class FlashBlock:
    def __init__(self, blockId, indices, bb, temp, tempdust, dens, vels, mags):
        self.id = blockId
        self._Ix, self._Iy, self._Iz = indices

        self.gridpoints = self.gridpointsForBoundingbox(bb)
        self.hullpoints = self.hullpoints()
        self.temperatures = self.temperatures(temp)
        self.dusttemperatures = self.dusttemperatures(tempdust)
        self.densities = self.densities(dens)
        self.velocities = self.velocities(vels)
        self.magfluxes = self.magfluxes(mags)

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
        1 indicating hullpoint, 0 indicating inner point"""
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
        return temperatures.flatten()

    def dusttemperatures(self, dusttemperatures):
        return dusttemperatures.flatten()

    def densities(self, densities):
        return densities.flatten()

    def velocities(self, velocities):
        return flatten3DValues(velocities[0], velocities[1], velocities[2])

    def magfluxes(self, magfluxes):
        return flatten3DValues(magfluxes[0], magfluxes[1], magfluxes[2])
