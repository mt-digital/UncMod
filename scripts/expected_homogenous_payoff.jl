using Dates
using DataFrames
using Distributed
using JLD2

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

function expected_social_payoff(env_uncertainty = collect(0.0:0.1:1.0); ntrials = 10,
                                low_payoff = [0.1,0.45,0.8], nbehaviors = [2,4], 
                                steps_per_round_vec = [1,2,4,8]
    )
    adata = [(:behavior, countmap), (:social_learner, mean), 
             (:prev_net_payoff, mean)]

    mdata = [:env_uncertainty, :trial_idx, :high_payoff, 
             :low_payoff, :nbehaviors, :steps_per_round, :optimal_behavior] 
    
    d = Dict{Int, DataFrame}()
    trial_idx = collect(1:ntrials)
    for steps_per_round in steps_per_round_vec

        params_list = dict_list(@dict env_uncertainty low_payoff nbehaviors steps_per_round trial_idx)

        L = steps_per_round
        println("Running $L steps per round")

        models = [
            uncertainty_learning_model(
                nagents = 100; init_social_learner_prevalence = 1.0, 
                params...
            )
            for params in params_list
        ]

        maxits = L * 100  # run 100 generations. Payoffs should stabilize by then.

        adf, mdf = ensemblerun!(models, agent_step!, model_step!, maxits;
                                adata, mdata, 
                                when = (_, step) -> step % 2L == 0,
                                parallel = true,
                                batch_size = max(length(models) ÷ 2nprocs(), 1))

        d[L] = innerjoin(adf, mdf, on = [:ensemble, :step])
    end
    
    return vcat(values(d)...)
end


function expected_individual_payoff(;
        low_payoff = [0.1,0.45,0.8], nbehaviors = [2,4], 
        steps_per_round_vec = [1,2,4,8], 
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
                                when = 
                                    (model, step) -> ( 
                                        (step % L == 0)  #||  
                                        # (step == 0)
                                ), 
                                parallel = true)

        d[L] = innerjoin(adf, mdf, on = [:ensemble, :step])
    end
    
    return vcat(values(d)...)
end


function all_expected_individual_payoffs(; savefile = "expected_individual.jld2")
    df_B_2_4 = expected_individual_payoff()
    df_B10 = expected_individual_payoff(;nbehaviors = [10], 
                                         steps_per_round_vec = [1,5,10,20])

    ret_df = vcat(df_B_2_4, df_B10)
    ret_df = ret_df[ret_df.step .!== 0, 
              [:low_payoff, :nbehaviors, :steps_per_round, :mean_prev_net_payoff]]
                                     
    if !isnothing(savefile)
        @save savefile ret_df
    end
    
    return ret_df
end
