using Test

using DrWatson
quickactivate("../../")
include("../model.jl")

@testset "Model should be initialized as specified" begin
    nteachers = 5
   
    learningstrategies = Dict(
        Group1 => LearningStrategy(0.25, 0.5, 0.75, 10),
        Group2 => LearningStrategy(0.33, 0.66, 1.0, 10)
    )

    environment = Dict(
        Behavior1 => BehaviorPayoffStructure(2.0, 0.5, 0.45),
        Behavior2 => BehaviorPayoffStructure(1.0, 0.25, 0.35)
    )

    model = uncertainty_learning_model(nagents = 10; 
                                       nteachers = nteachers,
                                       initlearningstrategies = learningstrategies,
                                       environment = environment)

    env_beh1 = model.properties[:environment][Behavior1]
    @test env_beh1.high_payoff == 2.0
    @test env_beh1.low_state_frac == 0.5
    @test env_beh1.reliability == 0.45

    env_beh2 = model.properties[:environment][Behavior2]
    @test env_beh2.high_payoff == 1.0
    @test env_beh2.low_state_frac == 0.25
    @test env_beh2.reliability == 0.35

    @test nagents(model) == 10
    @test all(map(a -> a.learning_strategy.nteachers, allagents(model)) .== 5)
end


@testset "Payoffs should have specified statistics." begin
    reliability = 1.0
    high_payoff = 2.0
    low_state_frac = 1.0
    paystruct = BehaviorPayoffStructure(high_payoff, low_state_frac, reliability)
    numpayoffs = 10_000.0
    @test sum([generate_payoff(paystruct) for _ in 1:numpayoffs]) == 20_000.0

    high_payoff = 4.0
    reliability = 0.5
    low_state_frac = 0.5
    paystruct = BehaviorPayoffStructure(high_payoff, low_state_frac, reliability)
    @test (sum([generate_payoff(paystruct) for _ in 1:numpayoffs]) - 30_000.0) / 30_000.0 < 1e-2

    low_state_frac = 0.0
    paystruct = BehaviorPayoffStructure(high_payoff, low_state_frac, reliability)
    @test (sum([generate_payoff(paystruct) for _ in 1:numpayoffs]) - 20_000.0) / 20_000.0 < 1e-2
end



@testset "Teachers should be selected in predictable ways." begin

    nteachers = 5
    model = uncertainty_learning_model(numagents = 20)

    @testset "Agents with homophily 1 should only select in-group teachers" begin
        
        focal_agent = model[1]
        focal_agent.learning_strategy = LearningStrategy(homophily = 1.0, 
                                                         nteachers = 5)
        # First agent is always in Group1.
        @assert focal_agent.group == Group1
            
        teachers = select_teachers(focal_agent, model)

        @test length(teachers) == nteachers

        # println(teachers)
        @test all(map(t -> t.group, teachers) .== Group1)
    end

end
