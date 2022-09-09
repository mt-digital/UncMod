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

function expected_social_payoff(;env_uncertainty = collect(0.0:0.1:1.0), ntrials = 10,
                                low_payoff = [0.1,0.45,0.8], nbehaviors = [2,4], 
                                steps_per_round = [1,2,4,8], tau = 0.1,
                                high_payoff = [0.9]
    )
    adata = [(:behavior, countmap), (:social_learner, mean), 
             (:prev_net_payoff, mean)]

    mdata = [:env_uncertainty, :trial_idx, :high_payoff, 
             :low_payoff, :nbehaviors, :steps_per_round, :optimal_behavior] 
    
    d = Dict{Int, DataFrame}()
    trial_idx = collect(1:ntrials)
    for L in steps_per_round

        params_list = dict_list(
            @dict env_uncertainty low_payoff nbehaviors steps_per_round trial_idx tau high_payoff
        )

        println("Running $L steps per round")

        models = [
            uncertainty_learning_model(
                nagents = 1000; 
                init_social_learner_prevalence = 1.0, 
                params...
            )
            for params in params_list
        ]

        maxits = L * 100  # run 100 generations. Payoffs should stabilize by then.
        batch_size = max(length(models) ÷ nprocs(), 1)
        adf, mdf = ensemblerun!(models, agent_step!, model_step!, maxits;
                                adata, mdata, 
                                when = (_, step) -> step % L == 0,
                                parallel = true,
                                batch_size
                               )

        d[L] = innerjoin(adf, mdf, on = [:ensemble, :step])
    end
    
    return vcat(values(d)...)
end


function expected_asocial_payoff(;
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
                nagents = 1000; init_social_learner_prevalence = 0.0, 
                params...
            )
            for params in params_list
        ]

        adf, mdf = ensemblerun!(models, agent_step!, model_step!, 10000*L;
                                adata, mdata, 
                                when = 
                                    (model, step) -> ( 
                                        (step % L == 0)  # ||  
                                        # (step == 0)
                                ), 
                                parallel = true)
        # return adf, mdf
        joined_df = innerjoin(adf, mdf, on = [:ensemble, :step])
        filter!(r -> r.step > 0, joined_df)
        gb = groupby(joined_df, [:ensemble, :env_uncertainty, :low_payoff,
                                 :nbehaviors, :steps_per_round])

        d[L] = combine(gb, :mean_prev_net_payoff => geomean => :geomean_payoff)

    end
    
    return vcat(values(d)...)
end


function all_expected_asocial_payoffs(; savefile = "expected_asocial.jld2")

    df_B_2_4 = expected_asocial_payoff()
    df_B10 = expected_asocial_payoff(;nbehaviors = [10], 
                                         steps_per_round_vec = [1,5,10,20])

    df = vcat(df_B_2_4, df_B10)
                                     
    if !isnothing(savefile)
        @save savefile df
    end

    println("saved")

    return df
end


function all_expected_social_payoffs(; savefile = "expected_social.jld2")
    for B in [2, 4, 10]
        aggdf = aggregate_final_timestep(
             load_expected_social_df(B), :mean_prev_net_payoff
        )
        @save "expected_social_B=$B.jld2" aggdf
    end
end


function load_expected_social_df(nbehaviors::Int; datadir = "data/expected_social", 
                                    jld2_key = "expected_social_joined_df"
    )

    if nbehaviors == 10
        filepaths_10 = glob("$datadir/*nbehaviors=[[]10*") 
        dfs = Vector{DataFrame}()
        ensemble_offset = 0
        
        for f in filepaths_10

            tempdf = load(f)[jld2_key]
            tempdf.ensemble .+= ensemble_offset
            ensemble_offset = maximum(tempdf.ensemble)

            push!(dfs, tempdf)
        end

        df10 = vcat(dfs...) 
        
        return df10
    else
        dfs = Vector{DataFrame}()
        ensemble_offset = 0

        filepaths_2_4 = glob("$datadir/*nbehaviors=[[]2,4[]]*")
        for f in filepaths_2_4

            tempdf = load(f)[jld2_key]
            filter!(r -> r.nbehaviors == nbehaviors, tempdf)

            tempdf.ensemble .+= ensemble_offset
            ensemble_offset = maximum(tempdf.ensemble)

            push!(dfs, tempdf)
        end

        return vcat(dfs...)
    end
end
                            

