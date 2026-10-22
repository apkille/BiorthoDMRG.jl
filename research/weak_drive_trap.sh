#!/bin/bash
#SBATCH --partition=genx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=weak_drive_trap
#SBATCH --mem=10GB
#SBATCH --array=1-30
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=apk9546@nyu.edu

OUTPUT_DIR="/mnt/ceph/users/akille/steady_state_data"

mkdir -p $OUTPUT_DIR

export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun julia --project=. weak_drive_trap.jl $OUTPUT_DIR $SLURM_ARRAY_TASK_ID