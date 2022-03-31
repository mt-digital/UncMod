using Test

using DrWatson
quickactivate("../../")
include("../model.jl")


@testset "Individual and social learners should be initialized as expected between generations" begin

    nbehaviors = 10
    steps_per_round = 5
    
    model = 
        uncertainty_learning_model(; τ = 1.0, nagents = 10, 
                                   nteachers = 9,
                                   nbehaviors = nbehaviors)

    # Set up model to have one agent to reproduce with much greater payoffs
    # and pre-defined ledger values.
    model[1].net_payoff = Inf
    model[1].ledger = rand(nbehaviors)
    model.tick = steps_per_round

    for agent in allagents(model)
        agent.social_learner = false
    end

    @testset "Next-gen individual learners should have zeroed-out ledger." begin

        _, _ = run!(model, agent_step!, model_step!, 1)   

        for agent in allagents(model)
            @test all(agent.ledger .== 0.0)
        end

    end


    for agent in allagents(model)
        agent.social_learner = true
    end

    model[1].net_payoff = Inf
    model[1].ledger = rand(nbehaviors)
    model.tick = steps_per_round

    @testset "Next-gen social learners should have best-performing ledger." begin
        _, _ = run!(model, agent_step!, model_step!, 1)
        for agent in allagents(model)
            @test agent.ledger == model[1].ledger
        end
    end
end


@testset "Agent should obtain payoffs with mean payoffs and behavior counts as expected when payoffs acquired fully randomly" begin

    high_payoff = 0.9
    nbehaviors = 10
    steps_per_round = 10000

    for low_payoff in 0.1:0.1:0.8

        model = 
            uncertainty_learning_model(; τ = Inf, nagents = 2, 
                                       nteachers = 1,
                                       low_payoff = low_payoff, 
                                       high_payoff = high_payoff, 
                                       steps_per_round = steps_per_round,
                                       nbehaviors = nbehaviors)

        _, _ = run!(model, agent_step!, model_step!, steps_per_round - 1)   

        a = model[1]
        bmax = model.optimal_behavior

        # Check all low-payoff ledger entries match model's low expected payoff.
        for b_idx in filter(b -> b ≠ bmax, 1:nbehaviors)
            @test a.ledger[b_idx] ≈ low_payoff atol=0.05
        end
        
        # Check high-payoff ledger entry matches model's high expected payoff.
        @test a.ledger[bmax] ≈ high_payoff atol=0.05

        # Check expected behaviors have been performed for each round step.
        # If we go full steps_per_round the behavior count would be reset to zero.
        @test sum(a.behavior_count) == steps_per_round - 1
    end
    
end


@testset "Vectorized softmax weight calculation should be as expected" begin
    
    payoffs = [1.0, 2.0, 3.0, 5.0]
    τ = 0.01

    denom = exp(1.0 / τ) + exp(2.0 / τ) + exp(3.0 / τ) + exp(5.0 / τ)
    w1 = exp(1.0 / τ) / denom
    w2 = exp(2.0 / τ) / denom
    w3 = exp(3.0 / τ) / denom
    w4 = exp(5.0 / τ) / denom
    w = [w1, w2, w3, w4]

    @test w == softmax(payoffs, τ)

end
