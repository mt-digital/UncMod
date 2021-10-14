using Dates
using DataFrames

include("model.jl")


function basic_demo_experiment(; nsteps = 1000, whensteps = 10, ntrials = 10)

    reliability_variances = [0.2, 0.15, 0.1, 5e-2, 1e-2, 1e-3, 1e-6]

    adata = adata = [(:soclearnfreq, mean)]

    # results = Dict{String,DataFrame}()
    results = DataFrame()

    base_reliabilities = [0.4 for _ in 1:10]
    base_reliabilities[6] = 0.6
    for rvar in reliability_variances

        for trial_idx in 1:ntrials
            then = now()

            model = uncertainty_learning_model(; 
                        nagents = 100, 
                        reliability_variance = rvar,
                        base_reliabilities = base_reliabilities,
                        steps_per_round = 20,
                        mutation_magnitude = 0.025)

            adf, mdf = run!(model, agent_step!, model_step!, nsteps;
                            adata, when = (model, s) -> s % whensteps == 0)

            adf[!, :trial] .= trial_idx
            adf[!, :relvar] .= string(rvar)

            results = vcat(results, adf)

            trialstime = Dates.toms(now() - then) / 1000.0

            println(
                "Ran trial $trial_idx for reliability variance $rvar in $trialstime secs"
            )
        end

        # results[string(rvar)] = adf_full
    end

    return results

end
