#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output=expectedpayoffs_vertical=true.out
#SBATCH --partition=serc,normal


module --force purge
module load devel
module load julia/1.6.2

julia scripts/run_trials.jl expected-payoff CogSci2022_Jan2 --high_payof=0.1:0.1:0.9 --low_payoff=0.1:0.1:0.9 --nbehaviors=20 --steps_per_round=50 --niter=100000 --ntrials=100 --vertical
