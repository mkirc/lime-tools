import h5py
import numpy as np
from helper import setupLIMEStage1, centerAxis


class FileWriter:
    def __init__(self, filePath):
        self.filePath = filePath

    def write(self, data):
        raise NotImplementedError


class LIMEWriter(FileWriter):
    def __init__(self, filePath, nBlocks):
        super().__init__(filePath)
        self.nBlocks = nBlocks

    def write(self, blocks):
        with h5py.File(f"{self.filePath}", "w") as outFile:
            radius = 2.7002512991143854e18
            minscale = 6.750628247785964e17

            x1, x2, x3, sinks = setupLIMEStage1(outFile, self.nBlocks, radius, minscale)

            i = 0
            for block in blocks:
                block.gridpoints[:, 0] = centerAxis(block.gridpoints[:, 0])
                block.gridpoints[:, 1] = centerAxis(block.gridpoints[:, 1])
                block.gridpoints[:, 2] = centerAxis(block.gridpoints[:, 2])

                x1[i * 512 : (i + 1) * 512] = block.gridpoints[:, 0]
                x2[i * 512 : (i + 1) * 512] = block.gridpoints[:, 1]
                x3[i * 512 : (i + 1) * 512] = block.gridpoints[:, 2]

                sinkpoints = np.take(block.gridpoints, np.nonzero(block.hullpoints), 0)[
                    0
                ]

                num_sinkpoints = len(sinkpoints)
                # num_innerpoints = len(block.gridpoints) - num_sinkpoints

                x1[(i + 1) * 512 : (i + 1) * 512 + num_sinkpoints] = sinkpoints[:, 0]
                x2[(i + 1) * 512 : (i + 1) * 512 + num_sinkpoints] = sinkpoints[:, 1]
                x3[(i + 1) * 512 : (i + 1) * 512 + num_sinkpoints] = sinkpoints[:, 2]

                sinks[i * 512 : (i + 1) * 512] = [0 for i in range(512)]
                sinks[(i + 1) * 512 : (i + 1) * 512 + num_sinkpoints] = [
                    1 for i in range(num_sinkpoints)
                ]

                i += 1
                del block


class CArrayWriter(FileWriter):
    def __init__(self, filePath, nBlocks):
        super().__init__(filePath)

        self.nBlocks = nBlocks

    def writeSingleBlock(self, block, radius=0, minscale=0):
        with open(f"{self.filePath}", "w") as outFile:
            size = self.nBlocks * 512
            lines = [f"static int model_size={size};"]
            lines += [f"static double model_radius={radius};"]
            lines += [f"static double model_minscale={minscale};"]
            lines += [
                f"static double model_x[{size}] = "
                + "{"
                + f"{','.join([str(p) for p in block.gridpoints[:, 0]])}"
                + "};"
            ]
            lines += [
                f"static double model_y[{size}] = "
                + "{"
                + f"{','.join([str(p) for p in block.gridpoints[:, 1]])}"
                + "};"
            ]
            lines += [
                f"static double model_z[{size}] = "
                + "{"
                + f"{','.join([str(p) for p in block.gridpoints[:, 2]])}"
                + "};"
            ]
            lines += [
                f"static double model_density[{size}] = "
                + "{"
                + f"{','.join([str(d) for d in block.densities])}"
                + "};"
            ]
            outFile.write("\n".join(lines))
