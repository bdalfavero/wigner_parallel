#!/bin/bash

#SBATCH --array=0-9
#SBATCH --time=02:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=intel18
#SBATCH --job-name=convergence
#SBATCH --mail-user=dalfaver@msu.edu
#SBATCH --output=job_array_%A_%a.out

case_files=($(ls -d *.h5))

export OMP_NUM_THREADS=4
exec_path=/mnt/home/dalfaver/wigner_parallel
my_file=${case_files[$SLURM_ARRAY_TASK_ID]}
echo $my_file
mpiexec -n 16 ${exec_path}/wigner ${my_file}
