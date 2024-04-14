#!/usr/bin/env python3

import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("output", help="Name of ouput file.", default="data_file.h5")
args = parser.parse_args()

f = h5py.File(args.output, 'w')

num_points_dset = f.create_dataset("num_points", (1,), dtype='i')
num_points_dset[0] = 50

num_samples_dset = f.create_dataset("num_samples", (1,), dtype='i')
num_samples_dset[0] = 500

time_step_group = f.create_group("time_steps")
time_step_dset = time_step_group.create_dataset("dt", (1,), dtype='double')
time_step_dset[0] = 1e-4
num_step_dset = time_step_group.create_dataset("num_steps", (1,), dtype='i')
num_step_dset[0] = 100

bose_group = f.create_group("bose_model")
g_dset = bose_group.create_dataset("g", (1,), dtype="double")
g_dset[0] = 1.0
t_dset = bose_group.create_dataset("t", (1,), dtype="double")
t_dset[0] = 1.0