# Run 1000 trials for both softmax sensitivity analyses.
sbatch --array=1-90 scripts/slurm/tau_sensitivity/B\=2_tau\=0.01.slurm
sbatch --array=1-90 scripts/slurm/tau_sensitivity/B\=4_tau\=0.01.slurm
sbatch --array=1-90 scripts/slurm/tau_sensitivity/B\=10_tau\=0.01.slurm
sbatch --array=1-90 scripts/slurm/tau_sensitivity/B\=2_tau\=1.0.slurm
sbatch --array=1-90 scripts/slurm/tau_sensitivity/B\=4_tau\=1.0.slurm
sbatch --array=1-90 scripts/slurm/tau_sensitivity/B\=10_tau\=1.0.slurm
