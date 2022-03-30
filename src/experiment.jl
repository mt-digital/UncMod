using Dates
using DataFrames


using Distributed

# Set up multiprocessing.
try
    num_cores = parse(Int, ENV["SLURM_CPUS_PER_TASK"])
    addprocs(num_cores)
catch
    desired_nprocs = length(Sys.cpu_info())

    if length(procs()) != desired_nprocs
        addprocs(desired_nprocs - 1)
    end
end


@everywhere using DrWatson
@everywhere quickactivate("..")
@everywhere include("model.jl")


function experiment(ntrials = 20; 
                    nagents = 100, 
                    nbehaviors = [2,3,5], #,10,20],
                    high_payoff = [0.9],  # π_high in the paper
                    # low_payoff = collect(0.1:0.1:0.89),   # π_low in the paper
                    low_payoff = [0.1, 0.5, 0.89],   # π_low in the paper
                    niter = 1000, 
                    # ngenerations = 20, 
                    steps_per_round = [1,2,5,10], # ,20],
                    mutation_prob = 0.05, 
                    whensteps = 10,
                    env_uncertainty = [0.0, 0.33, 0.66, 1.0])
    
    trial_idx = collect(1:ntrials)

    params_list = dict_list(
        @dict steps_per_round nbehaviors high_payoff low_payoff trial_idx env_uncertainty mutation_prob
    )

    # We are not interested in cases where high expected payoff is less than
    # or equal to the lower expected payoff, only cases where high expected
    # payoff is greater than low expected payoff.
    params_list = filter(
        params -> params[:high_payoff] > params[:low_payoff],
        params_list
    )

    countbehaviors(behaviors) = countmap(behaviors)

    adata = [(:behavior, countbehaviors), (:social_learner, mean)]
    mdata = [:env_uncertainty, :optimal_behavior, :trial_idx, :high_payoff, :low_payoff, :nbehaviors, :steps_per_round] 

    models = [
        uncertainty_learning_model(;
            nagents = nagents, 
            params...)

        for params in params_list
    ]
    
    # niter = ngenerations * steps_per_round

    return ensemblerun!(
        models, agent_step!, model_step!, niter; 
        adata, mdata, 
        when = (model, step) -> ( (step + 1) % whensteps == 0  ||  step == 0 ),
        # when = (step) -> ( (step + 1) % whensteps == 0  ||  step == 0 ),
        parallel = true
    )
end


