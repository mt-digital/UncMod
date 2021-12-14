include("run_trials.jl")

tic = now()

run_trials(10; 
           steps_per_round = 100,
           nbehaviors = collect(5:5:100),
           niter = 100_000, transledger = false, 
           outputfilename = "data/firstdraft/nbehaviors_10trials_notrans_100stepsper.jld2")

trialstime = Dates.toms(now() - tic) / (60.0 * 1000.0)

println("Ran expected payoffs trials in $trialstime minutes")
