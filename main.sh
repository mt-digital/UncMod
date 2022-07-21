# Run 1000 trials for the main "expected social" experiment (TODO: better name)
sbatch --array=1-10 scripts/slurm/expected_social/B\=2.slurm
sbatch --array=1-10 scripts/slurm/expected_social/B\=4.slurm
sbatch --array=1-10 scripts/slurm/expected_social/B\=10.slurm
