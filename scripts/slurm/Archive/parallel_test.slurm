#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:01:00
#SBATCH --output=run_trials.out
#SBATCH --partition=serc,normal


module --force purge
module load julia

julia scripts/slurm/run_trials.jl
