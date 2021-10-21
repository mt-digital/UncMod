using Dates
using DataFrames

include("model.jl")


function prepare_parameters(
        ntrials, 
        reliability_variances, nrounds, steps_per_round, whensteps, nbehaviors,
        mutation_magnitude, regen_reliabilities
    )

    trial_idx = collect(1:ntrials)

    return @dict reliability_variances steps_per_round nbehaviors mutation_magnitude regen_reliabilities softmax_exploration
end

function experiment(ntrials = 100; 
                    nagents = 100, reliability_variances = [0.15, 1e-6], 
                    niter = 1e5, steps_per_round = 100, nbehaviors = 5,
                    mutation_magnitude = 0.1, regen_reliabilities = false,
                    softmax_exploration = 4.0)
end

function basic_demo_experiment(; 
    reliability_variances = [0.01, 1e-6], 
    nagents = 100,
    nsteps = 1000, whensteps = 10, ntrials = 10, steps_per_round = 100,
    mutation_magnitude = 0.1, regen_reliabilities = true, nbehaviors = 10,
    softmax_exploration = 1.0)

    
    mean_soclearn(soclearnfreq) = mean(soclearnfreq)
    modal_behavior(behaviors) = findmax(countmap(behaviors))
    adata = adata = [(:behavior, modal_behavior), (:soclearnfreq, mean)]

    results = DataFrame()

    base_reliabilities = [0.2 for _ in 1:nbehaviors]
    base_reliabilities[1] = 0.8
    
    for rvar in reliability_variances

        for trial_idx in 1:ntrials

            then = now()

            model = uncertainty_learning_model(; 
                        nagents = nagents, 
                        reliability_variance = rvar,
                        base_reliabilities = base_reliabilities,
                        steps_per_round = steps_per_round,
                        mutation_magnitude = mutation_magnitude,
                        regen_reliabilities = regen_reliabilities,
                        softmax_exploration = softmax_exploration)

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
