include("run_trials.jl")

run_trials(2; 
           steps_per_round = 50,
           nbehaviors = collect(5:5:100),
           niter = 2_000, transledger = true, 
           outputfilename = "data/firstdraft/nbehaviors.jld2")
