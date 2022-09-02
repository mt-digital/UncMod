# Run 1000 trials for the main experiment
sbatch --array=1-100 scripts/slurm/full_B\=2.slurm
sbatch --array=1-100 scripts/slurm/full_B\=4.slurm
sbatch --array=1-100 scripts/slurm/full_B\=10.slurm
