import pathlib

import h5py
import numpy as np

from flashBlock import FlashFactory
# from writer import CArrayWriter
from helper import centerAxis, sampleSpherical, radiusForBoundingboxes
from limefile import LimeFile


def allBlocksTest():
    testDir = pathlib.Path(__file__).parent.absolute()
    plotFile = testDir.joinpath("be_hdf5_plt_cnt_0004")
    outFile = testDir.joinpath("all_test.h5")
    NSINKS = 5000

    with LimeFile(f"{str(outFile)}", "w") as limeFile:
        flashFile = h5py.File(plotFile, "r")

        ff = FlashFactory(flashFile)

        # nBlocks = len(ff.leaves)
        nBlocks = 1
        allLeafSlice = slice(0, nBlocks)

        print(ff.radius)
        print(ff.minscale)

        # generate sinkpoints
        sinkpoints = sampleSpherical(NSINKS) * ff.radius

        # prepare outfile
        limeFile.setupStageOne(
            nBlocks=nBlocks, nSinks=NSINKS, radius=ff.radius, minscale=ff.minscale
        )

        # write data
        limeFile.writeStageOne(ff.generateBlocksForSlice(allLeafSlice), sinkpoints)


def singleBlockTest():
    testDir = pathlib.Path(__file__).parent.absolute()
    plotFile0 = testDir.joinpath("be_hdf5_plt_cnt_0004")
    singleBlockOutFile = testDir.joinpath("test.h5")
    NSINKS = 1000

    with LimeFile(f"{str(singleBlockOutFile)}", "w") as limeFile:
        flashFile = h5py.File(plotFile0, "r")

        ff = FlashFactory(flashFile)

        block = next(ff.generateBlocksForSlice(slice(0, 1)))

        # block.gridpoints[:, 0] = centerAxis(block.gridpoints[:, 0])
        # block.gridpoints[:, 1] = centerAxis(block.gridpoints[:, 1])
        # block.gridpoints[:, 2] = centerAxis(block.gridpoints[:, 2])

        radius = np.sqrt(
            np.max(np.abs(block.gridpoints[:, 0])) ** 2
            + np.max(np.abs(block.gridpoints[:, 1])) ** 2
            + np.max(np.abs(block.gridpoints[:, 2])) ** 2
        )

        sinkpoints = sampleSpherical(NSINKS) * radius * 100

        limeFile.setupStageOne(
            nBlocks=1, nSinks=NSINKS, radius=ff.radius, minscale=ff.minscale
        )

        limeFile.writeStageOne([block], sinkpoints)


# def singleBlockCArrayTest():
#     testDir = pathlib.Path(__file__).parent.absolute()
#     plotFile0 = testDir.joinpath("SpitzerTest_hdf5_plt_cnt_0000")
#     singleBlockOutFile = testDir.joinpath("model_constants.h")

#     flashFile = h5py.File(plotFile0, "r")

#     ff = FlashBlockFactory(flashFile)
#     cw = CArrayWriter(str(singleBlockOutFile), nBlocks=1)

#     block = next(ff.generateBlocksForSlice(slice(0, 1)))

#     block.gridpoints[:, 0] = centerAxis(block.gridpoints[:, 0])
#     block.gridpoints[:, 1] = centerAxis(block.gridpoints[:, 1])
#     block.gridpoints[:, 2] = centerAxis(block.gridpoints[:, 2])

#     radius = np.max(block.gridpoints[:, 0])
#     minscale = 2 * np.max(block.gridpoints[:, 0]) / 8

#     cw.writeSingleBlock(block, radius, minscale)
