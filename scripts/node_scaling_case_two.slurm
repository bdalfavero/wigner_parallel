#!/bin/bash

#SBATCH --array=0-4
#SBATCH --time=01:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=intel18
#SBATCH --job-name=node_scaling
#SBATCH --mail-user=dalfaver@msu.edu
#SBATCH --output=job_array_%A_%a.out

export OMP_NUM_THREADS=1
nnodes=(1 2 4 8 16)
my_nnodes=${nnodes[$SLURM_ARRAY_TASK_ID]}
exec_path=/mnt/home/dalfaver/wigner_parallel
my_file=out_case_two_${my_nnodes}.h5
cp out_case_two.h5 ${my_file}
mpiexec -n ${my_nnodes} ${exec_path}/wigner ${my_file}
