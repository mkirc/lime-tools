import h5py
from helper import setupLIMEStage1


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
        with h5py.File(f"{filePath}", "w") as outFile:
            x1, x2, x3 = setupLIMEStage1(outFile, self.nBlocks)

            i = 0
            for block in blocks:
                x1[i * 512 : (i + 1) * 512] = block.gridpoints[:, 0]
                x2[i * 512 : (i + 1) * 512] = block.gridpoints[:, 1]
                x3[i * 512 : (i + 1) * 512] = block.gridpoints[:, 2]
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
