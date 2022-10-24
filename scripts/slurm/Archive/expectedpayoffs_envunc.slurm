#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=10G
#SBATCH --time=00:00:30
#SBATCH --output=testfull.out
#SBATCH --partition=serc,normal


module --force purge
module load devel
module load julia/1.6.2

julia scripts/run_trials.jl expected-payoff cogsci2022-firstdraft --env_uncertainty=1.0 --high_payof=0.1:0.1:0.9 --low_payoff=0.1:0.1:0.9 --nbehaviors=100 --steps_per_round=20 --niter=100000 --ntrials=100 --vertical
