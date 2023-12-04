import sys
import argparse

import h5py
from flashBlock import FlashFactory
from limeFile import LimeFile
from helper import (
    centerAxis,
    sampleSphereSurface,
    radiusForBoundingboxes,
    sampleSphere,
    createArgumentParser,
)


def parseArgs():
    argParser = createArgumentParser()

    if len(sys.argv) < 2:
        arg_parser.print_help()
        exit()

    args = arg_parser.parse_args()

    return args


def convertWithArgs(args):
    with LimeFile(f"{str(args.outFile)}", "w") as limeFile:
        flashFile = h5py.File(args.inFile, "r")

        ff = FlashFactory(flashFile)

        nBlocks = args.blocks if args.blocks is not None else len(ff.leaves)
        nSinks = args.sinks if args.sinks is not None else 1000
        radscale = args.radscale if args.radscale is not None else 1

        # print(ff.radius)
        # print(ff.minscale)

        # generate sinkpoints
        sinkpoints = sampleSphereSurface(nSinks) * ff.radius * radscale

        # prepare outfile
        limeFile.setupFileAttributes(radius=ff.radius * radscale, minscale=ff.minscale)
        limeFile.setupPrimaryGroups()
        limeFile.setupPoints(nBlocks=nBlocks, nSinks=nSinks)

        # prepare properties
        limeFile.setupDensity()
        limeFile.setupGasTemperature()
        limeFile.setupVelocity()
        limeFile.setupMagfield()

        # write data
        allLeafSlice = slice(0, nBlocks)
        limeFile.writeBlocks(ff.generateBlocksForSlice(allLeafSlice))
        limeFile.writeSinks(sinkpoints)


if __name__ == "__main__":
    args = parseArgs()

    convertWithArgs(args)
