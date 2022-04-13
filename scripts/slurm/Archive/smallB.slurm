#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --output=smallB.out
#SBATCH --partition=serc,normal


module --force purge
module load devel
module load julia/1.6.2

julia scripts/run_trials.jl smallB smallB --env_uncertainty=[0.0,0.25,0.5,0.75,1.0] --high_payof=[0.2,0.9] --low_payoff=[0.1] --nbehaviors=[2,3] --steps_per_round=[2,3,5,10,20] --niter=100000 --ntrials=10 --vertical
