#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:01:00
#SBATCH --output=parallel_test.out
#SBATCH --partition=serc,normal


module --force purge
module load julia

julia parallel_test.jl
