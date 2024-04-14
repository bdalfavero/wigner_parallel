#!/bin/bash

 export OMP_NUM_THREADS=10

./scripts/prepare_input.py out.h5
./wigner out.h5 > output.csv
./scripts/plot_population.py output.csv -o population.png
./scripts/get_avg_time.py output.csv