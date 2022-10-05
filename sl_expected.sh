# Run 1000 trials for the main "expected social" experiment.
sbatch --array=1-100 scripts/slurm/social_learners_expected/full_B\=2.slurm
sbatch --array=1-100 scripts/slurm/social_learners_expected/full_B\=4.slurm
sbatch --array=1-100 scripts/slurm/social_learners_expected/full_B\=10.slurm
