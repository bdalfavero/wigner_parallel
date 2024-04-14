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

times = df.loc[:, df.columns == "walltime"].values

print(np.mean(times))