using Test

using DrWatson
quickactivate("../../")
include("../model.jl")

# @testset "Model should be initialized as specified" begin
#     nteachers = 5
   
#     learningstrategies = Dict(
#         Group1 => LearningStrategy(0.25, 0.5, 0.75, 10),
#         Group2 => LearningStrategy(0.33, 0.66, 1.0, 10)
#     )

#     environment = Dict(
#         Behavior1 => BehaviorPayoffStructure(2.0, 0.5, 0.45),
#         Behavior2 => BehaviorPayoffStructure(1.0, 0.25, 0.35)
#     )

#     model = uncertainty_learning_model(nagents = 20; 
#                                        nteachers = nteachers,
#                                        initlearningstrategies = learningstrategies,
#                                        environment = environment,
#                                        R0_1 = 0.5,
#                                        R0_2 = 0.25)

#     env_beh1 = model.properties[:environment][Behavior1]
#     @test env_beh1.high_payoff == 2.0
#     @test env_beh1.low_state_frac == 0.5
#     @test env_beh1.reliability == 0.45

#     env_beh2 = model.properties[:environment][Behavior2]
#     @test env_beh2.high_payoff == 1.0
#     @test env_beh2.low_state_frac == 0.25
#     @test env_beh2.reliability == 0.35

#     @test nagents(model) == 20
#     @test all(map(a -> a.learning_strategy.nteachers, allagents(model)) .== 5)

#     # # Check that correct number of Group1 agents init'd to Behavior1.
#     @test count(a -> a.behavior == Behavior1, 
#                 filter(a -> a.group == Group1, collect(allagents(model)))
#                 ) == 5
    
#     # Check that correct number of Group2 agents init'd to Behavior2.
#     @test count(a -> a.behavior == Behavior2, 
#                 filter(a -> a.group == Group2, collect(allagents(model)))
#                 ) == 3
# end


# @testset "Payoffs should have specified statistics." begin
#     reliability = 1.0
#     high_payoff = 2.0
#     low_state_frac = 1.0
#     paystruct = BehaviorPayoffStructure(high_payoff, low_state_frac, reliability)
#     numpayoffs = 10_000
#     @test sum([generate_payoff(paystruct) for _ in 1:numpayoffs]) == 20_000.0

#     high_payoff = 4.0
#     reliability = 0.5
#     low_state_frac = 0.5
#     paystruct = BehaviorPayoffStructure(high_payoff, low_state_frac, reliability)
#     @test (sum([generate_payoff(paystruct) for _ in 1:numpayoffs]) - 30_000.0) / 30_000.0 < 1e-2

#     low_state_frac = 0.0
#     paystruct = BehaviorPayoffStructure(high_payoff, low_state_frac, reliability)
#     @test (sum([generate_payoff(paystruct) for _ in 1:numpayoffs]) - 20_000.0) / 20_000.0 < 1e-2
# end



# @testset "Teachers should be selected in predictable ways." begin

#     nteachers = 5
#     model = uncertainty_learning_model(numagents = 20)

#     @testset "Agents with homophily 1 should only select in-group teachers" begin
        
#         focal_agent = model[1]
#         focal_agent.learning_strategy = LearningStrategy(homophily = 1.0, 
#                                                          nteachers = 5)
#         # First agent is always in Group1.
#         @assert focal_agent.group == Group1
            
#         teachers = select_teachers(focal_agent, model)

#         @test length(teachers) == nteachers

#         @test all(map(t -> t.group, teachers) .== Group1)
#     end

#     @testset "Agents with homophily 0 should select in-group teachers half the time." begin
        
#         focal_agent = model[1]
#         focal_agent.learning_strategy = LearningStrategy(homophily = 1.0, 
#                                                          nteachers = 5)
#         # First agent is always in Group1.
#         @assert focal_agent.group == Group1
            
#         teachers = select_teachers(focal_agent, model)

#         @test length(teachers) == nteachers

#         @test all(map(t -> t.group, teachers) .== Group1)
#     end
# end

# @testset "Social and asocial learning should select teachers and behaviors as expected according to learner and model parameters" begin
#     @testset "Social learners should be conformist learners, i.e., select the most common strategy among teachers." begin
        
#         # social_learning_only_strategy = LearningStrategy(soclearnfreq = 1.0)
#         # learner = LearningAgent(group=Group1, behavior=Behavior1, 
#         #                         learning_strategy=social_learning_only_strategy)

#         nagents = 11
#         model = uncertainty_learning_model(; nagents = nagents)

#         # for (agent_idx, _) in enumerate(allagents(model))

#         for agent_idx in 1:nagents
#             if agent_idx < 8
#                 model[agent_idx].behavior = Behavior1
#             else
#                 model[agent_idx].behavior = Behavior2 
#             end
#         end
#         # print(allagents(model))

#         learner = model[1]
#         learner.learning_strategy = LearningStrategy(soclearnfreq = 1.0, 
#                                                      homophily = 0.0,
#                                                      exploration=0.0)

#         ntrials = 1000
#         select_behavior_trials = [select_behavior!(learner, model) 
#                                   for _ in 1:ntrials]
#         # println(select_behavior_trials)

#         @test all(Behavior1 .== select_behavior_trials)

#         for agent_idx in 1:nagents
#             if agent_idx > 8
#                 model[agent_idx].behavior = Behavior1
#             else
#                 model[agent_idx].behavior = Behavior2 
#             end
#         end

#         select_behavior_trials = [select_behavior!(learner, model) 
#                                   for _ in 1:ntrials]

#         @test all(Behavior2 .== select_behavior_trials)

#         learner.learning_strategy = LearningStrategy(soclearnfreq = 1.0, 
#                                                      homophily = 0.5,
#                                                      exploration=0.5)

#         ntrials = 10000
#         select_behavior_trials = [select_behavior!(learner, model) 
#                                   for _ in 1:ntrials]
        
#         countBeh2 = count(res -> res == Behavior2, select_behavior_trials)
         
#         expectedCountBeh2 = (ntrials / 2) + (ntrials / 4)

#         # println(countBeh2)
#         relerr = abs(countBeh2 - expectedCountBeh2) / expectedCountBeh2
#         # println(relerr)
#         @test relerr < 1e-2
#     end


#     @testset "Selected teachers should be unique" begin
#         nagents = 11
#         model = uncertainty_learning_model(; nagents = nagents)
#         learner = model[1]

#         ntrials = 1000

#         teachers_trials = [select_teachers(learner, model) for _ in ntrials]

#         @test all(map(teachers -> length(teachers) == length(unique(teachers)),
#                       teachers_trials)) 
#     end

#     @testset "Asocial learning should select the behavior with the best average historical payoffs with probability ϵ." begin

#         nagents = 11
#         model = uncertainty_learning_model(; nagents = nagents)
#         learner = model[1]

#         learner.learning_strategy = LearningStrategy(exploration = 0.0, 
#                                                      soclearnfreq = 0.0)


#         learner.ledger = Dict(Behavior1 => 100.55, Behavior2 => 100.50)

#         ntrials = 100
#         select_behavior_trials = [select_behavior!(learner, model) 
#                                   for _ in 1:ntrials]

#         @test all(map(beh -> beh == Behavior1, select_behavior_trials))
        
        
#         # Test non-zero exploration with Behavior1 superior.
#         learner.learning_strategy = LearningStrategy(exploration = 0.5, 
#                                                      soclearnfreq = 0.0)

#         ntrials = 10_000

#         select_behavior_trials = [select_behavior!(learner, model) 
#                                   for _ in 1:ntrials]
        
#         countBeh1 = count(res -> res == Behavior1, select_behavior_trials)
         
#         expectedCountBeh1 = (ntrials / 2) + (ntrials / 4)

#         relerr = abs(countBeh1 - expectedCountBeh1) / expectedCountBeh1

#         @test relerr < 1e-2


#         # Zero exploration with Behavior2 superior.
#         learner.ledger = Dict(Behavior1 => 10.12, Behavior2 => 100.50)

#         learner.learning_strategy = LearningStrategy(exploration = 0.0, 
#                                                      soclearnfreq = 0.0)

#         ntrials = 100
#         select_behavior_trials = [select_behavior!(learner, model) 
#                                   for _ in 1:ntrials]

#         @test all(map(beh -> beh == Behavior2, select_behavior_trials))

#         # Non-zero exploration with Behavior2 superior.
#         learner.learning_strategy = LearningStrategy(exploration = 0.5, 
#                                                      soclearnfreq = 0.0)

#         ntrials = 10_000
#         select_behavior_trials = [select_behavior!(learner, model) 
#                                   for _ in 1:ntrials]
        
#         countBeh2 = count(res -> res == Behavior2, select_behavior_trials)
         
#         expectedCountBeh2 = (ntrials / 2) + (ntrials / 4)

#         relerr = abs(countBeh2 - expectedCountBeh2) / expectedCountBeh2

#         @test relerr < 1e-2
#     end
# end


@testset "Reproduction and die-off should work as expected for special cases" begin

    model = uncertainty_learning_model(; nagents = 10, ntoreprodie = 5, )

    earners_payoff = 1000.0
    foreach(ii -> model[ii].net_payoff = earners_payoff, 1:5)

    reproducers = select_reproducers(model)

    terminals_age = 1000
    foreach(ii -> model[ii].age = terminals_age, 5:10)

    terminals = select_todie(model)

    @testset "Reproduction should favor those with highest profits and record parent-child relationships" begin

        repro_ids = map(r -> r.id, reproducers)
        expected_in_repro_ids = map(ii -> ii ∈ 1:5, repro_ids)
        @test all(expected_in_repro_ids)
    end

    # @testset "The correct number of old-ass agents should be selected to die off" begin
    #     @test all(map(r -> r.id, terminals) .∈ 5:10)
    # end

    # @testset "Parent-child relationships should be accurately recorded" begin
    #     evolve!(model)
    #     @test all(
    #         map(r -> r.parent, model[5:10]) .∈ map(r -> r.uuid, model[1:5])
    #     )
    # end
end
