#!/bin/bash

#SBATCH --array=0-4
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --constraint=intel18
#SBATCH --job-name=node_scaling
#SBATCH --mail-user=dalfaver@msu.edu
#SBATCH --output=job_array_%A_%a.out

export OMP_NUM_THREADS=1
nnodes=(1 2 4 8 16)
my_nnodes=${nnodes[$SLURM_ARRAY_TASK_ID]}
exec_path=/mnt/home/dalfaver/wigner_parallel
my_file=same_node_${my_nnodes}.h5
cp out.h5 ${my_file}
mpiexec -n ${my_nnodes} ${exec_path}/wigner ${my_file}
