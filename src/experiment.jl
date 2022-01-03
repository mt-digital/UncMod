using Dates
using DataFrames


using Distributed

try
    num_cores = parse(Int, ENV["SLURM_CPUS_PER_TASK"])
    addprocs(num_cores)
catch
end


@everywhere using DrWatson
@everywhere quickactivate("..")
@everywhere include("model.jl")





function experiment(ntrials = 100; 
                    nagents = 100, 
                    reliability_variance = [1e-8], 
                    nbehaviors = [5, 20, 100],
                    high_payoff = [0.2, 0.9],  # π_high in the paper
                    low_payoff = [0.1, 0.8],   # π_low in the paper
                    niter = 10_000, 
                    steps_per_round = 100,
                    mutation_magnitude = 0.05, 
                    regen_reliabilities = true,
                    vertical = true,
                    whensteps = 1_000,
                    env_uncert = 0.0)
    
    trial_idx = collect(1:ntrials)

    params_list = dict_list(
        @dict reliability_variance steps_per_round nbehaviors high_payoff low_payoff trial_idx vertical
    )

    # We are not interested in cases where high expected payoff is less than
    # or equal to the lower expected payoff, only cases where high expected
    # payoff is greater than low expected payoff.
    params_list = filter(
        params -> params[:high_payoff] > params[:low_payoff],
        params_list
    )

    countbehaviors(behaviors) = countmap(behaviors)

    adata = [(:behavior, countbehaviors), (:soclearnfreq, mean), (:vertical_squeeze, mean)]
    mdata = [:reliability_variance, :trial_idx, :high_payoff, :low_payoff, :nbehaviors, :steps_per_round] 

    models = [
        uncertainty_learning_model(;
            nagents = nagents, 
            params...)

        for params in params_list
    ]

    return ensemblerun!(
        models, agent_step!, model_step!, niter; 
        adata, mdata, 
        when = (model, step) -> ( (step + 1) % whensteps == 0  ||  step == 0 ),
        parallel = true
    )
end


# function makeresdf(adf, mdf, models)
#     res = mdf[!, [:high_payoff, :low_payoff, :nbehaviors]]
#     res.pct_optimal = map(r -> r.countbehaviors_behavior[1] / length(models[1].agents), eachrow(adf))

#     res[!, :steps] .= adf.step[1]
    
#     return res
# end


# function basic_demo_experiment(; 
#     reliability_variances = [0.01, 1e-6], 
#     nagents = 100,
#     nsteps = 1000, whensteps = 10, ntrials = 10, steps_per_round = 100,
#     mutation_magnitude = 0.1, regen_reliabilities = true, nbehaviors = 10,
#     τ_init = 1.0, low_payoff = 0.4, high_payoff = 0.6)

    
#     mean_soclearn(soclearnfreq) = mean(soclearnfreq)
#     # modal_behavior(behaviors) = findmax(countmap(behaviors))
#     # adata = adata = [(:modal_behavior, modal_behavior), (:soclearnfreq, mean)]
#     countbehaviors(behaviors) = countmap(behaviors)
#     adata = adata = [(:behavior, countbehaviors), (:soclearnfreq, mean)]

#     results = DataFrame()

#     base_reliabilities = [low_payoff for _ in 1:nbehaviors]
#     base_reliabilities[1] = high_payoff
    
#     for rvar in reliability_variances

#         for trial_idx in 1:ntrials 

#             then = now() 

#             model = uncertainty_learning_model(; 
#                         nagents = nagents,  
#                         reliability_variance = rvar,
#                         base_reliabilities = base_reliabilities,
#                         steps_per_round = steps_per_round,
#                         mutation_magnitude = mutation_magnitude,
#                         regen_reliabilities = regen_reliabilities,
#                         selection_strategy = selection_strategy,
#                         τ_init = τ_init)

#             adf, mdf = run!(model, agent_step!, model_step!, nsteps;
#                             adata, when = (model, s) -> s % whensteps == 0)

#             adf[!, :trial] .= trial_idx
#             adf[!, :reliability_var] .= string(rvar)

#             results = vcat(results, adf)

#             trialstime = Dates.toms(now() - then) / 1000.0

#             println(
#                 "Ran trial $trial_idx for reliability variance $rvar in $trialstime secs"
#             )

#         end
#     end

#     return results
# end

# function individual_learning_pilot(
#         nagents = 100;
#         nbehaviors = [2, 5, 10, 20, 50, 100], 
#         reliability_variance = [1e-6, 1e-3, 1e-1],
#         high_payoff = [0.2, 0.6, 0.9], low_payoff = [0.1, 0.5, 0.8],
#         steps = 20, selection_strategy = ϵGreedy, selection_temperature = 0.05
#     )

#     params_list = dict_list(
#         @dict reliability_variance steps nbehaviors high_payoff low_payoff  
#     )

#     params_list = filter(
#         params -> params[:high_payoff] > params[:low_payoff],
#         params_list
#     )

#     countbehaviors(behaviors) = countmap(behaviors)
#     # noptimal(behaviors) = countmap(behaviors)[1] / nagents
#     # ledgers(ledger) = mean(ledger)
#     # adata = [(:behavior, countbehaviors), (:behavior, noptimal)]
#     # adata = [(:behavior, countbehaviors), (:ledger, ledgers)]
#     adata = [(:behavior, countbehaviors)]
#     # adata = [:behavior]
#     mdata = [:reliability_variance, :high_payoff, :low_payoff, :nbehaviors] 

#     if selection_strategy == ϵGreedy
#         indivlearn_models = [
#             uncertainty_learning_model(;
#                 nagents = nagents, steps_per_round = steps + 1, 
#                 selection_strategy = selection_strategy, 
#                 ϵ_init = selection_temperature, params...)
#             for params in params_list
#         ]
#     else
#         indivlearn_models = [
#             uncertainty_learning_model(;
#                 nagents = nagents, steps_per_round = steps + 1, 
#                 selection_strategy = selection_strategy, 
#                 τ_init = selection_temperature, params...)
#             for params in params_list
#         ]
#     end
    
#     return ensemblerun!(
#         indivlearn_models, agent_step!, model_step!, steps; 
#         adata, mdata, when = ((model, step) -> step == steps)
#     )
# end


