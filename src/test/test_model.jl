using Test

using DrWatson
quickactivate("../../")
include("../model.jl")



@testset "Payoffs should have specified statistics." begin
    
    @testset "Reliabilities should have expected statistics" begin

        base_rels = [0.2, 0.8]
        rel_var = 1e-8
        model = uncertainty_learning_model(; 
            n_agents=10, base_reliabilities=base_rels, reliability_variance=rel_var
        ) 
        @test all(
            map(agent -> all(
                map(((idx, rel),) -> isapprox(rel, base_rels[idx]; 
                                              atol=1e-4, rtol=1e-2),
                    enumerate(agent.reliabilities))),
                allagents(model)
            )
        )
        @test all(agent -> length(agent.reliabilities) == 2, allagents(model))

        base_rels = [0.4, 0.4, 0.8]
        rel_var = 1e-4
        model = uncertainty_learning_model(; 
            n_agents=10, base_reliabilities=base_rels, reliability_variance=rel_var
        ) 
        @test all(
            map(agent -> all(
                map(((idx, rel),) -> isapprox(rel, base_rels[idx]; 
                                              atol=1e-4, rtol=1e-1),
                    enumerate(agent.reliabilities))),
                allagents(model)
            )
        )
        @test all(agent -> length(agent.reliabilities) == 3, allagents(model))

    end

    @testset "Payoffs based on set reliabilities should have specified statistics" begin
        model = uncertainty_learning_model(; n_agents=10) 

        agent = model[1]

        r1 = 0.1; r2 = 0.45; r3 = 0.8
        agent.reliabilities = [r1 r2 r3]
        
        agent.behavior = 1
        payoffs = sum(map(_ -> generate_payoff!(agent), 1:1000))
        @test isapprox(payoffs, 100.0; rtol=1e-1)

        agent.behavior = 2
        payoffs = sum(map(_ -> generate_payoff!(agent), 1:1000))
        @test isapprox(payoffs, 450.0; rtol=1e-1)

        agent.behavior = 3
        payoffs = sum(map(_ -> generate_payoff!(agent), 1:1000))
        @test isapprox(payoffs, 800.0; rtol=1e-1)
    end

end

@testset "Social and asocial learning should select behaviors as expected according to learner and model parameters" begin

    model = uncertainty_learning_model(; nagents = 11, nteachers = 5, )

    focal_agent = model[1]
    focal_agent.behavior = 1
    focal_agent.soclearnfreq = 1.0

    # Set all non-focal agent behaviors to 2.
    foreach(a -> a.behavior = 2, 
            Iterators.filter(a -> a.id ≠ 1, allagents(model)))

    # Make sure focal agent selects behavior 2.
    select_behavior!(focal_agent, model)
    @test focal_agent.behavior == 2

end


@testset "Reproduction and die-off should work as expected for special cases" begin

    model = uncertainty_learning_model(; nagents = 10, ntoreprodie = 5, )

    earners_payoff = 1000.0
    foreach(ii -> model[ii].net_payoff = earners_payoff, 1:5)

    reproducers = select_reproducers(model)

    terminals_age = 1000
    foreach(ii -> model[ii].age = terminals_age, 6:10)

    terminals = select_to_die(model, reproducers)

    evolve!(model)

    sorted_agents = sort(collect(allagents(model)), by = agent -> agent.id)

    @testset "Reproduction should favor those with highest profits and record parent-child relationships." begin

        repro_ids = map(r -> r.id, reproducers)
        expected_in_repro_ids = map(rid -> rid ∈ 1:5, repro_ids)
        @test all(expected_in_repro_ids)
    end

    @testset "The correct number of old-ass agents should be selected to die off." begin
        terminal_ids = map(t -> t.id, terminals)
        expected_in_terminal_ids = map(tid -> tid ∈ 6:10, terminal_ids)
        @test all(expected_in_terminal_ids)
    end

    @testset "Parent-child relationships should be accurately recorded." begin

        expected_children = sorted_agents[6:10]
        @test all(map(child -> !isnothing(child.parent), expected_children))

        expected_children_parents = map(r -> r.parent, expected_children)
        expected_parents_uuids = map(parent -> parent.uuid, sorted_agents[1:5])
        @test all(
            map(parent -> parent ∈ expected_parents_uuids, 
                expected_children_parents)
        )
    end

    @testset "Inherited learning parameter should have changed through drift." begin

        parent_by_uuid = Dict(
            parent.uuid => parent
            for parent in sorted_agents[1:5]
        )

        @test all(
            map(
                child -> 
                    child.soclearnfreq ≠ 
                    parent_by_uuid[child.parent].soclearnfreq,
                sorted_agents[6:10]
            )
        )
    end
end

