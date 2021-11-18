using Distributed

using DrWatson
quickactivate("..")

using JLD2

include("../../src/experiment.jl")

# Launch worker processes.
# num_cores = parse(Int, ENV["SLURM_CPUS_PER_TASK"])
# addprocs(num_cores)
# addprocs(3)

adf, mdf, models = 
    experiment(100; 
        nbehaviors=[5, 20, 100], niter=100_000, 
    whensteps = 1000);

adf.pct_optimal = map(
    r -> (haskey(r.countbehaviors_behavior, 1) ? 
            r.countbehaviors_behavior[1] : 
            0.0 )  / length(models[1].agents), 
    eachrow(adf)
)

resdf = innerjoin(adf[!, Not([:countbehaviors_behavior])], 
                  mdf, 
                  on = [:ensemble, :step])

resdf_trialsmean = combine(
    # Groupby experimental variables...
    groupby(resdf, [:step, :nbehaviors, :low_reliability, 
            :high_reliability, :reliability_variance]),

    # ...and aggregate by taking means over outcome variables, convert to table.
    [:mean_soclearnfreq, :pct_optimal] 
        =>
            (
                (soclearnfreq, pct_optimal) -> 
                    (soclearnfreq = mean(soclearnfreq),
                     pct_optimal = mean(pct_optimal))
            ) 
        =>
            AsTable
)

resdf_trialsmean.baserels_nbehs = 
    map(r -> string((r.high_reliability, r.low_reliability, r.nbehaviors)), 
        eachrow(resdf_trialsmean))

@save "trials_output.jld2" resdf_trialsmean
