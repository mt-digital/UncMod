@testset "Softmax behavior selection should select behaviors at random when τ = Inf" begin
    

    nbehaviors = 10
    low_payoff = 0.1
    high_payoff = 0.9 
    model = uncertainty_learning_model(; τ = Inf, nagents = 2, 
                                       nteachers = 1,
                                       low_payoff, 
                                       high_payoff, 
                                       steps_per_round = 10,
                                       nbehaviors) 

    ntrials = Int(1e6)

    @test model.expected_payoffs[model.optimal_behavior] == 0.9
    for ii in 1:nbehaviors
        if ii ≠ model.optimal_behavior
            @test model.expected_payoffs[ii] == 0.1
        end
    end

    beh_count = zeros(nbehaviors)
    payoffs = zeros(nbehaviors)
    focal_agent = model[1]
    for _ in 1:ntrials
        select_behavior!(focal_agent, model)
        add_step_payoff!(focal_agent, model)
        beh_count[focal_agent.behavior] += 1
        payoffs[focal_agent.behavior] += focal_agent.step_payoff
    end

    for ii in 1:nbehaviors
        @test beh_count[ii] ≈ (ntrials / nbehaviors) rtol=0.05
        if ii == model.optimal_behavior
            @test payoffs[ii] ≈ (high_payoff * (ntrials/nbehaviors)) rtol=0.05
        else
            @test payoffs[ii] ≈ (low_payoff * (ntrials/nbehaviors)) rtol=0.05
        end
    end
        
end
