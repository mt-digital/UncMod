#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=10G
#SBATCH --time=00:30:00
#SBATCH --output=expected_social_B=2,4.out
#SBATCH --partition=serc,normal


module --force purge
module load devel
module load julia/1.7.2

julia scripts/run_expected_social_payoff.jl expected_social --ntrials=100 --env_uncertainty=0.0:0.1:1.0 --low_payoff=\[0.1,0.45,0.8\] --high_payoff=\[0.9\]
