#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=10G
#SBATCH --time=00:10:00
#SBATCH --output=full_B=2.out
#SBATCH --partition=serc,normal


module --force purge
module load devel
module load julia/1.7.2

julia scripts/run_trials.jl main --max_niter=100000 --ntrials=10 --env_uncertainty=0.0:0.1:1.0 --nbehaviors=\[2\] --steps_per_round=\[1,2,4,8\] --low_payoff=\[0.1,0.45,0.8\] --high_payoff=\[0.9\] --numagents=1000 --stop_cond=fixation_plus_onegen
