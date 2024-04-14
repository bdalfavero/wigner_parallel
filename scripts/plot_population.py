#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input filename")
parser.add_argument("-o", "--output", default="population.png")
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

df = pd.read_csv(args.input)
population = df.loc[:, df.columns != "t"].values

fig, ax = plt.subplots()
ax.imshow(population, aspect="auto")
#fig.set_figheight(3.0)
#fig.set_figwidth(3.0)
if args.show:
    plt.show()
plt.savefig(args.output)