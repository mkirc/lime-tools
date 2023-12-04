import csv
import pathlib

import numpy as np
from matplotlib import pyplot as plt

outDir = pathlib.Path(__file__).parent.absolute() / "out"
inDir = pathlib.Path(__file__).parent.absolute() / "in"


tbars = []

fig, ax = plt.subplots()

for nThreads in range(1, 7):
    nBlocks = []
    convertT = []
    execT = []
    success = []

    csvFilePath = inDir.joinpath(f"execT_{nThreads}_threads.csv")
    with open(str(csvFilePath), "r") as csvFile:
        reader = csv.reader(csvFile)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            nBlocks.append(int(row[0]))
            convertT.append(float(row[1]))
            execT.append(float(row[2]))
            success.append(bool(row[3]))

    coeff = np.polyfit(nBlocks[3:], execT[3:], 1)

    tbars.append(coeff[0])

    xFit = np.linspace(nBlocks[3] - 10, nBlocks[-1] + 10, 50)
    yFit = xFit * coeff[0] + coeff[1]

    ax.plot(
        nBlocks,
        execT,
        linestyle=":",
        marker=".",
        color="grey",
    )
    ax.plot(
        xFit,
        yFit,
        "-",
        linewidth=1,
        label=(
            f"{nThreads} Thread{'s' if nThreads > 1 else ''}, "
            "($\overline{t}$ "
            f"= {coeff[0]:.3})"
        ),
    )
    ax.plot(nBlocks, convertT, "--.")

ax.set_xlabel("Number of Blocks")
ax.set_ylabel("Elapsed time [s]")
plt.legend()
plt.savefig(str(outDir.joinpath("execTimes.pdf")), bbox_inches='tight')

fig, ax = plt.subplots()

ax.plot(
    [i for i in range(1,len(tbars)+1)],
    tbars,
    linestyle=":",
    marker=".",
    color="grey",
    label="s/Block/Thread"
)

xFit = np.linspace(1, 6, 6)
yFit = (1/xFit) * 5.92

# damp = ( np.log(tbars[-1]) - np.log(tbars[-1]) ) / 5
# eFit = np.exp(-(xFit - 0) * damp) *  5.92
# ax.plot(xFit, eFit)

ax.plot(xFit, yFit, linewidth=1, label="1/n")

ax.set_xlabel("Number of Threads")
ax.set_ylabel("Elapsed time per Block [s/Block]")
plt.legend()
plt.savefig(str(outDir.joinpath("tBars.pdf")), bbox_inches='tight')
