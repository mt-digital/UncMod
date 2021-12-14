using Distributed

using DrWatson
quickactivate("..")

using Comonicon
using JLD2

include("../../src/experiment.jl")


function run_trials(ntrials = 100; 
                    outputfilename = "trials_output.jld2", 
                    experiment_kwargs...)

    adf, mdf, models = experiment(ntrials; experiment_kwargs...)

    adf.pct_optimal = map(
        r -> (haskey(r.countbehaviors_behavior, 1) ? 
                r.countbehaviors_behavior[1] : 
                0.0 )  / length(models[1].agents), 
        eachrow(adf)
    )

    resdf = innerjoin(adf[!, Not([:countbehaviors_behavior])], 
                      mdf, 
                      on = [:ensemble, :step])

    result = combine(
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

    result.baserels_nbehs = 
        map(r -> string((r.high_reliability, r.low_reliability, r.nbehaviors)), 
            eachrow(result))
    

    @save outputfilename result

end

# run_trials(; niter = 100_000, outputfilename = "softmax.jld2")
