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


function experiment(ntrials = 10; 
                    nagents = 100, 
                    nbehaviors = [2], #,10,20],
                    high_payoff = [0.9],  # π_high in the paper
                    low_payoff = [0.1, 0.5, 0.8],   # π_low in the paper
                    niter = 1000, 
                    steps_per_round = [1,2,5],
                    whensteps = 100,
                    env_uncertainty = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
                    # env_uncertainty = collect(0.0:0.1:1.0)
    )
    
    trial_idx = collect(1:ntrials)

    params_list = dict_list(
        @dict steps_per_round nbehaviors high_payoff low_payoff trial_idx env_uncertainty
    )

    # We are not interested in cases where high expected payoff is less than
    # or equal to the lower expected payoff, only cases where high expected
    # payoff is greater than low expected payoff.
    params_list = filter(
        params -> params[:high_payoff] > params[:low_payoff],
        params_list
    )

    adata = [(:behavior, countmap), (:social_learner, mean)]
    mdata = [:env_uncertainty, :trial_idx, :high_payoff, :low_payoff, :nbehaviors, :steps_per_round, :optimal_behavior] 

    models = [
        uncertainty_learning_model(;
            nagents = nagents, 
            params...)

        for params in params_list
    ]
    
    adf, mdf = ensemblerun!(
        models, agent_step!, model_step!, niter; 
        adata, mdata, 
        when = (model, step) -> ( (step + 1) % whensteps == 0  ||  step == 0 ),
        # when = (step) -> ( (step + 1) % whensteps == 0  ||  step == 0 ),
        parallel = true
    )

    println("About to return adf, mdf!!!")

    return adf, mdf
end


