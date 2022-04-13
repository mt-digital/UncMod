#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00
#SBATCH --output=full.out
#SBATCH --partition=serc,normal


module --force purge
module load devel
module load julia/1.6.2

julia scripts/run_trials.jl develop --ntrials=100 --env_uncertainty=0.0:0.1:1.0 --nbehaviors=\[4\] --steps_per_round=\[2,4,8\] --low_payoff=\[0.1,0.5,0.8\] --high_payoff=\[0.9\]
