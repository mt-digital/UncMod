#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --output=nbehaviors.out
#SBATCH --partition=serc,normal


module --force purge
module load devel
module load julia/1.6.2

julia scripts/run_trials.jl nbehaviors CogSci2022_Jan2 --high_payof=0.2,0.9 --low_payoff=0.1,0.8 --nbehaviors=10:10:100 --steps_per_round=50 --niter=100000 --ntrials=100 
