import pathlib

import h5py
import numpy as np

from flashBlock import FlashFactory
from helper import centerAxis, sampleSphereSurface, radiusForBoundingboxes, sampleSphere
from limeFile import LimeFile

from copy import deepcopy


def allBlocksTest():
    testDir = pathlib.Path(__file__).parent.absolute() / "tests"
    plotFile = testDir.joinpath("in/be_hdf5_plt_cnt_0102")
    outFile = testDir.joinpath("out/all_test.h5")
    nSinks = 1000

    with LimeFile(f"{str(outFile)}", "w") as limeFile:
        flashFile = h5py.File(plotFile, "r")

        ff = FlashFactory(flashFile)

        nBlocks = len(ff.leaves)
        # nBlocks = 1
        allLeafSlice = slice(0, nBlocks)
        radscale = 0.1

        # print(ff.radius)
        # print(ff.minscale)

        # generate sinkpoints
        sinkpoints = sampleSphereSurface(nSinks) * ff.radius * radscale

        # prepare outfile
        limeFile.setupFileAttributes(radius=ff.radius * radscale,
                                     minscale=ff.minscale)
        limeFile.setupPrimaryGroups()
        limeFile.setupPoints(nBlocks=nBlocks, nSinks=nSinks)

        # prepare properties
        limeFile.setupDensity()
        limeFile.setupGasTemperature()
        limeFile.setupVelocity()

        # write data
        limeFile.writeBlocks(ff.generateBlocksForSlice(allLeafSlice))
        limeFile.writeSinks(sinkpoints)


def threeBlocksTest():
    testDir = pathlib.Path(__file__).parent.absolute() / "tests"
    plotFile0 = testDir.joinpath("in/be_hdf5_plt_cnt_0004")
    singleBlockOutFile = testDir.joinpath("out/three_test.h5")
    nSinks = 1000
    radScale = 1

    with LimeFile(f"{str(singleBlockOutFile)}", "w") as limeFile:
        flashFile = h5py.File(plotFile0, "r")

        ff = FlashFactory(flashFile)

        block = next(ff.generateBlocksForSlice(slice(0, 1)))

        block.gridpoints[:, 0] = centerAxis(block.gridpoints[:, 0])
        block.gridpoints[:, 1] = centerAxis(block.gridpoints[:, 1])
        block.gridpoints[:, 2] = centerAxis(block.gridpoints[:, 2])

        blockOne = deepcopy(block)
        blockOne.gridpoints[:, 0] += ff.radius * 0.2
        blockOne.gridpoints[:, 1] += ff.radius * 0.2
        blockOne.gridpoints[:, 2] += ff.radius * 0.2

        blockTwo = deepcopy(block)
        blockTwo.gridpoints[:, 0] -= ff.radius * 0.2
        blockTwo.gridpoints[:, 1] -= ff.radius * 0.2
        blockTwo.gridpoints[:, 2] -= ff.radius * 0.2

        sinkpoints = sampleSphereSurface(nSinks) * ff.radius * radScale

        blocks = [block, blockOne, blockTwo]

        limeFile.setupFileAttributes(radius=ff.radius * radScale,
                                     minscale=ff.minscale)
        limeFile.setupPrimaryGroups()

        limeFile.setupPoints(nBlocks=len(blocks), nSinks=nSinks)

        limeFile.writeBlocks(blocks)
        limeFile.writeSinks(sinkpoints)


def singleBlockTest():
    testDir = pathlib.Path(__file__).parent.absolute() / "tests"
    plotFile0 = testDir.joinpath("in/be_hdf5_plt_cnt_0004")
    singleBlockOutFile = testDir.joinpath("out/single_test.h5")
    nSinks = 1000
    radScale = 1

    with LimeFile(f"{str(singleBlockOutFile)}", "w") as limeFile:
        flashFile = h5py.File(plotFile0, "r")

        ff = FlashFactory(flashFile)
        print(ff.radius)
        print(ff.minscale)

        block = next(ff.generateBlocksForSlice(slice(0, 1)))

        block.gridpoints[:, 0] = centerAxis(block.gridpoints[:, 0])
        block.gridpoints[:, 1] = centerAxis(block.gridpoints[:, 1])
        block.gridpoints[:, 2] = centerAxis(block.gridpoints[:, 2])

        sinkpoints = sampleSphereSurface(nSinks) * ff.radius * radScale

        limeFile.setupFileAttributes(ff.radius * radScale,
                                     ff.minscale)
        limeFile.setupPrimaryGroups()
        limeFile.setupPoints(nBlocks=1, nSinks=nSinks)

        limeFile.setupDensity()
        limeFile.setupGasTemperature()
        limeFile.setupVelocity()

        limeFile.writeBlocks([block])
        limeFile.writeSinks(sinkpoints)


def suzanneTest():
    class DummyBlock:
        def __init__(self):
            self.gridpoints = None

    vertices = []

    with open("tests/in/suzanne.obj", "r") as tpFile:
        for line in tpFile:
            if line[0] == "v" and line[1] != "n":
                _, x, y, z = line.split()
                vertices.append([float(x), float(y), float(z)])
    print(f"{len(vertices)} vertices")

    scalefactor = 1e17
    nBlocks = len(vertices) // 512
    blocks = []

    for i in range(nBlocks):
        block = DummyBlock()
        block.gridpoints = np.array(vertices[i * 512 : (i + 1) * 512])
        # block.gridpoints[:, 1] -= 1.5
        block.gridpoints *= scalefactor
        print(block.gridpoints.shape)
        blocks.append(block)

    radius = np.sqrt(
        np.max(np.abs(block.gridpoints[:, 0])) ** 2
        + np.max(np.abs(block.gridpoints[:, 1])) ** 2
        + np.max(np.abs(block.gridpoints[:, 2])) ** 2
    )

    minscale = 1e-3 * radius

    print(radius)
    print(minscale)

    with LimeFile("tests/out/joke.h5", "w") as limeFile:
        nSinks = 1000
        sinkpoints = sampleSphereSurface(nSinks) * radius * 1.5
        limeFile.setupPointsAndSinks(
            nBlocks=len(blocks), nSinks=nSinks, radius=radius, minscale=minscale
        )
        limeFile.writeBlocksAndSinks(blocks, sinkpoints)


if __name__ == "__main__":
    singleBlockTest()
    allBlocksTest()
    # threeBlocksTest()
