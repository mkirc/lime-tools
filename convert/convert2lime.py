import sys

import h5py

from flashBlock import FlashBlockFactory
from writer import LIMEWriter


if __name__ == "__main__":
    N_BLOCKS = 3

    flashFile = h5py.File(sys.argv[1], "r")
    outFilePath = sys.argv[2]
    ff = FlashBlockFactory(flashFile)
    lw = LIMEWriter(outFilePath)
    lw.write(ff.generateBlocksForSlice(slice(0,N_BLOCKS)))
