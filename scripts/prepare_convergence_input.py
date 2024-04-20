#!/usr/bin/env python3

import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("min_samples", help="Minimum number of samples.")
parser.add_argument("max_samples", help="Maximum number of samples.")
parser.add_argument("n_experiments", help="Number of sample size")
args = parser.parse_args()

interval = int((int(args.max_samples) - int(args.min_samples)) / int(args.n_experiments))
num_samples = range(int(args.min_samples), int(args.max_samples), interval)

for i in range(len(num_samples)):

	f = h5py.File("convergence_{}.h5".format(num_samples[i]), 'w')

	num_points_dset = f.create_dataset("num_points", (1,), dtype='i')
	num_points_dset[0] = 50

	num_samples_dset = f.create_dataset("num_samples", (1,), dtype='i')
	num_samples_dset[0] = num_samples[i]

	time_step_group = f.create_group("time_steps")
	time_step_dset = time_step_group.create_dataset("dt", (1,), dtype='double')
	time_step_dset[0] = 1e-4
	num_step_dset = time_step_group.create_dataset("num_steps", (1,), dtype='i')
	num_step_dset[0] = 1000

	bose_group = f.create_group("bose_model")
	g_dset = bose_group.create_dataset("g", (1,), dtype="double")
	g_dset[0] = 1.0
	t_dset = bose_group.create_dataset("t", (1,), dtype="double")
	t_dset[0] = 50.0

	init_pop_dset = f.create_dataset("init_population", (1,), dtype="double")
	init_pop_dset[0] = 5.0

	simd_dset = f.create_dataset("use_simd", (1,), dtype="int")
	simd_dset[0] = 1
