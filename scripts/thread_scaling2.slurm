#!/bin/bash

#SBATCH --array=0-4
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=thread_scaling
#SBATCH --mail-user=dalfaver@msu.edu
#SBATCH --output=job_array_%A_%a.out
#SBATCH --error=job_array_%A_%a.err

nthreads=(1 2 4 8 16)
my_nthreads=${nthreads[$SLURM_ARRAY_TASK_ID]}
exec_path=/mnt/home/dalfaver/wigner_parallel
export OMP_NUM_THREADS=$my_nthreads
my_file=thread_scaling2_${my_nthreads}.h5
cp out.h5 ${my_file}
mpiexec -n 1 ${exec_path}/wigner ${my_file}
