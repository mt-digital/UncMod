include("run_trials.jl")

tic = now()

run_trials(100; 
           steps_per_round = 50,
           # nbehaviors = [5, 20, 100],
           nbehaviors = 20,
           high_reliability = collect(0.1:0.1:0.9),
           low_reliability =  collect(0.1:0.1:0.9),
           niter = 100_000, transledger = false, 
           outputfilename = "data/firstdraft/expectedpayoffs_transledger=false.jld2")

trialstime = Dates.toms(now() - tic) / (60.0 * 1000.0)

println("Ran expected payoffs trials in $trialstime minutes")
