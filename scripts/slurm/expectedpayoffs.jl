include("run_trials.jl")

run_trials(2; 
           steps_per_round = 50,
           nbehaviors = [5, 20, 100],
           high_reliability = collect(0.1:0.1:0.9),
           low_reliability =  collect(0.1:0.1:0.9),
           niter = 2_000, transledger = true, 
           outputfilename = "data/firstdraft/expectedpayoffs.jld2")
