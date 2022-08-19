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
                    low_payoff = [0.1, 0.45, 0.8],   # π_low in the paper
                    max_niter = 1000, 
                    steps_per_round = [1,2,4,8],
                    whensteps = 100,
                    env_uncertainty = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
                    random_init = false,
                    tau = 0.1,
                    nteachers = 5,
                    init_social_learner_prevalence = 0.5,
                    stop_cond = :default
                    # env_uncertainty = collect(0.0:0.1:1.0)
    )
    
    trial_idx = collect(1:ntrials)

    params_list = dict_list(
        @dict steps_per_round nbehaviors high_payoff low_payoff trial_idx env_uncertainty tau nagents nteachers init_social_learner_prevalence
    )

    # We are not interested in cases where high expected payoff is less than
    # or equal to the lower expected payoff, only cases where high expected
    # payoff is greater than low expected payoff.
    params_list = filter(
        params -> params[:high_payoff] > params[:low_payoff],
        params_list
    )

    adata = [(:behavior, countmap), (:social_learner, mean), 
             (:prev_net_payoff, mean)]

    mdata = [:env_uncertainty, :trial_idx, :high_payoff, 
             :low_payoff, :nbehaviors, :steps_per_round, 
             :optimal_behavior] 

    models = [
        uncertainty_learning_model(;
            nagents = nagents, 
            params...)

        for params in params_list
    ]

    function stop_condfn(model, step)
        n_sl = sum(a.social_learner for a in allagents(model))

        fixated = (n_sl == 0.0) || (n_sl == nagents)

        if stop_cond == :default
            return fixated || step > max_niter
        elseif stop_cond == :all_social_learners
            return step > max_niter * model.properties[:steps_per_round]
        end
    end

    adf, mdf = ensemblerun!(
        models, agent_step!, model_step!, stop_condfn; 
        adata, mdata, 
        when = (model, step) -> ( 
            ((step + 1) % whensteps == 0)  ||  (step == 0) || stop_condfn(model, step) 
        ),
        parallel = true,
        batch_size = max(length(models) ÷ nprocs(), 1)
    )

    println("About to return adf, mdf!!!")

    return adf, mdf
end


