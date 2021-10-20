using Dates
using DataFrames

include("model.jl")


function basic_demo_experiment(; 
    reliability_variances = [0.2, 0.15, 0.1, 5e-2, 1e-2, 1e-3, 1e-6], 
    nsteps = 1000, whensteps = 10, ntrials = 10, steps_per_round = 100,
    mutation_magnitude = 0.05, regen_reliabilities = true)

    
    mean_soclearn(soclearnfreq) = mean(soclearnfreq)
    modal_behavior(behaviors) = findmax(countmap(behaviors))
    adata = adata = [(:behavior, modal_behavior), (:soclearnfreq, mean)]

    results = DataFrame()

    base_reliabilities = [0.2 for _ in 1:5]
    base_reliabilities[5] = 0.8
    
    for rvar in reliability_variances

        for trial_idx in 1:ntrials

            then = now()

            model = uncertainty_learning_model(; 
                        nagents = 100, 
                        reliability_variance = rvar,
                        base_reliabilities = base_reliabilities,
                        steps_per_round = steps_per_round,
                        mutation_magnitude = mutation_magnitude,
                        regen_reliabilities = regen_reliabilities)

            adf, mdf = run!(model, agent_step!, model_step!, nsteps;
                            adata, when = (model, s) -> s % whensteps == 0)

            adf[!, :trial] .= trial_idx
            adf[!, :reliability_var] .= string(rvar)

            results = vcat(results, adf)

            trialstime = Dates.toms(now() - then) / 1000.0

            println(
                "Ran trial $trial_idx for reliability variance $rvar in $trialstime secs"
            )

        end
    end

    return results

end
