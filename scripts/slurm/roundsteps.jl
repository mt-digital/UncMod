include("run_trials.jl")

tic = now()

run_trials(100; 
           steps_per_round = [5, 10, 20, 50, 100],
           nbehaviors = [5, 20, 100],
           niter = 100_000, transledger = true, 
           outputfilename = "data/firstdraft/roundsteps.jld2")

trialstime = Dates.toms(now() - tic) / (60.0 * 1000.0)

println("Ran expected payoffs trials in $trialstime minutes")
