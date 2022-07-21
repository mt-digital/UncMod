# Run 1000 trials for N_T=2,10,20.
sbatch --array=1-10 scripts/slurm/nteachers_sensitivity/B\=2_nteachers\=2.slurm
sbatch --array=1-10 scripts/slurm/nteachers_sensitivity/B\=2_nteachers\=10.slurm
sbatch --array=1-10 scripts/slurm/nteachers_sensitivity/B\=2_nteachers\=20.slurm
sbatch --array=1-10 scripts/slurm/nteachers_sensitivity/B\=4_nteachers\=2.slurm
sbatch --array=1-10 scripts/slurm/nteachers_sensitivity/B\=4_nteachers\=10.slurm
sbatch --array=1-10 scripts/slurm/nteachers_sensitivity/B\=4_nteachers\=20.slurm
sbatch --array=1-10 scripts/slurm/nteachers_sensitivity/B\=10_nteachers\=2.slurm
sbatch --array=1-10 scripts/slurm/nteachers_sensitivity/B\=10_nteachers\=10.slurm
sbatch --array=1-10 scripts/slurm/nteachers_sensitivity/B\=10_nteachers\=20.slurm
