# Run 1000 trials for the main "expected social" experiment (TODO: better name)
sh main.sh

# Run 1000 trials for both softmax sensitivity analyses.
sh tau.sh

# Run 1000 trials for N=50,200 pop. sensitivity analyses; 
# 100 trials for N=1000 (set in .slurm scripts).
sh nagents.sh

# Run 1000 trials for N_T=2,10,20.
sh nteachers.sh
