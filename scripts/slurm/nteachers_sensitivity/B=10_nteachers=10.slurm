#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=10G
#SBATCH --time=01:00:00
#SBATCH --output=nteachers_sensitivity.out
#SBATCH --partition=serc,normal


module --force purge
module load devel
module load julia/1.7.2

julia scripts/run_trials.jl nteachers_sensitivity --max_niter=20000 --ntrials=10 --env_uncertainty=0.0:0.1:1.0 --nbehaviors=\[10\] --steps_per_round=\[1,5,10,20\] --low_payoff=\[0.1,0.45,0.8\] --high_payoff=\[0.9\] --nteachers=10 --nagents=1000
