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
@everywhere include("../src/model.jl")


function expected_individual_payoff(;
        low_payoff = [0.1,0.45,0.8], nbehaviors = [2,4], 
        steps_per_round_vec = [1,2,4,8]
    )
    

    adata = [(:behavior, countmap), (:social_learner, mean), 
             (:prev_net_payoff, mean)]

    mdata = [:env_uncertainty, :trial_idx, :high_payoff, 
             :low_payoff, :nbehaviors, :steps_per_round, :optimal_behavior] 
    
    d = Dict{Int, DataFrame}()
    for steps_per_round in steps_per_round_vec

        params_list = dict_list(@dict low_payoff nbehaviors steps_per_round)

        L = steps_per_round
        println("Running $L steps per round")

        models = [
            uncertainty_learning_model(
                nagents = 10000; init_social_learner_prevalence = 0.0, 
                params...
            )
            for params in params_list
        ]

        adf, mdf = ensemblerun!(models, agent_step!, model_step!, L;
                                adata, mdata, 
                                # when = 
                                #     (model, step) -> ( 
                                #         (step % L == 0)  ||  
                                #         (step == 0)
                                # ), 
                                parallel = true)

        d[L] = innerjoin(adf, mdf, on = [:ensemble, :step])
    end
    
    return d
end

function main()
    dexpected_individual_payoff
end
