import sys

import h5py

from flashBlock import FlashBlockFactory
from writer import CArrayWriter


if __name__ == "__main__":
    N_BLOCKS = 1

    flashFile = h5py.File(sys.argv[1], "r")
    outFilePath = sys.argv[2]
    ff = FlashBlockFactory(flashFile)
    cw = CArrayWriter(outFilePath, 1)
    cw.writeSingleBlock(ff.generateBlocksForSlice(slice(0, N_BLOCKS)))
