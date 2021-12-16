include("run_trials.jl")

tic = now()

run_trials(100; 
           steps_per_round = 100,
           nbehaviors = collect(10:10:100),
           niter = 100_000, 
           outputfilename = "data/firstdraft/nbehaviors_10to100behaviors.jld2")

trialstime = Dates.toms(now() - tic) / (60.0 * 1000.0)

println("Ran expected payoffs trials in $trialstime minutes")
