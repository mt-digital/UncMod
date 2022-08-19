# Run 1000 trials for N_T=2,10,20.
sbatch --array=1-10 scripts/slurm/N_sensitivity/B\=2_N\=50.slurm
sbatch --array=1-10 scripts/slurm/N_sensitivity/B\=2_N\=200.slurm
sbatch --array=1-10 scripts/slurm/N_sensitivity/B\=2_N\=1000.slurm
sbatch --array=1-10 scripts/slurm/N_sensitivity/B\=4_N\=50.slurm
sbatch --array=1-10 scripts/slurm/N_sensitivity/B\=4_N\=200.slurm
sbatch --array=1-10 scripts/slurm/N_sensitivity/B\=4_N\=1000.slurm
sbatch --array=1-10 scripts/slurm/N_sensitivity/B\=10_N\=50.slurm
sbatch --array=1-10 scripts/slurm/N_sensitivity/B\=10_N\=200.slurm
sbatch --array=1-10 scripts/slurm/N_sensitivity/B\=10_N\=1000.slurm
