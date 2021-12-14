include("run_trials.jl")

run_trials(2; 
           steps_per_round = [5, 20],
           nbehaviors = [5, 20, 100],
           niter = 10_000, transledger = true, 
           outputfilename = "data/firstdraft/roundsteps.jld2")
