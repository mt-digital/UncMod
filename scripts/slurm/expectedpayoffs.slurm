#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --output=expectedpayoffs.out
#SBATCH --partition=serc,normal


module --force purge
module load devel
module load julia/1.6.2

julia scripts/slurm/expectedpayoffs.jl
