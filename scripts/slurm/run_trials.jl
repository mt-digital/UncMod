using Distributed

using DrWatson
quickactivate("..")

using Comonicon
using JLD2

include("../../src/experiment.jl")

# Launch worker processes.
# num_cores = parse(Int, ENV["SLURM_CPUS_PER_TASK"])
# addprocs(num_cores)
# addprocs(3)

# <<<<<<< Updated upstream
# adf, mdf, models = 
#     experiment(100; 
#         nbehaviors=[5, 20, 100], niter=100_000, 
#     whensteps = 1000);

# adf.pct_optimal = map(
#     r -> (haskey(r.countbehaviors_behavior, 1) ? 
#             r.countbehaviors_behavior[1] : 
#             0.0 )  / length(models[1].agents), 
#     eachrow(adf)
# )

# resdf = innerjoin(adf[!, Not([:countbehaviors_behavior])], 
#                   mdf, 
#                   on = [:ensemble, :step])

# resdf_trialsmean = combine(
#     # Groupby experimental variables...
#     groupby(resdf, [:step, :nbehaviors, :low_reliability, 
#             :high_reliability, :reliability_variance]),

#     # ...and aggregate by taking means over outcome variables, convert to table.
#     [:mean_soclearnfreq, :pct_optimal] 
#         =>
#             (
#                 (soclearnfreq, pct_optimal) -> 
#                     (soclearnfreq = mean(soclearnfreq),
#                      pct_optimal = mean(pct_optimal))
#             ) 
#         =>
#             AsTable
# )

# resdf_trialsmean.baserels_nbehs = 
#     map(r -> string((r.high_reliability, r.low_reliability, r.nbehaviors)), 
#         eachrow(resdf_trialsmean))

function run_trials(ntrials = 100; outputfilename = "trials_output.jld2", experiment_kwargs...)

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
    
    
    
    # @save "$(now()).jld2" result adf mdf #  models

    @save outputfilename result

end

run_trials(; niter = 100_000, selection_strategy = Softmax, selection_temperature = 0.01, outputfilename = "softmax.jld2")
