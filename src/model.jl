##
# Model of the evolution of social learning under group membership and uncertain
# environmental conditions.
#
# Author: Matthew A. Turner
# Date: 2021-09-06
#

using Distributed
using Agents
using Distributions
using DrWatson
@everywhere using Parameters
using StatsBase
using UUIDs

using JuliaInterpreter
using Debugger


"""
Initialize one instance of the social learning model with the given parameters.

Notes: 
- we use tau in the model code here, not β as we did in the paper; Recall 
  β = 1/tau. 
"""
function uncertainty_learning_model(; 
                                    numagents = 100,  
                                    nbehaviors = 2,
                                    steps_per_round = 5,
                                    nteachers = 5,
                                    init_social_learner_prevalence = 0.5,
                                    tau = 0.1,
                                    high_payoff = 0.9,
                                    low_payoff = 0.1,
                                    trial_idx = nothing,
                                    env_uncertainty = 0.0,
                                    ngenerations_fixated = 1,
                                    model_parameters...)
    
    tick = 1
    # Optimal behavior will be randomly set between 1 and nbehaviors below.
    optimal_behavior = 0
    expected_payoffs = []

    # Set up tracking to possibly stop the model after ngenerations. 
    # `model.stop` will be set to true based on a check of whether the model
    # is fixated. `model.fixated` will be set last in model_step! after 
    # this check, so that model.stop is detected after one fixated generation.
    if ngenerations_fixated > 1
        @assert false "Maximum number of generations to run past fixation limited to 1"
    end
    fixated = false
    stop = false

    # Build full dictionary of model parameters and mutation distribution.
    params = merge(
        Dict(model_parameters),  

        @dict steps_per_round tick low_payoff high_payoff nbehaviors nteachers trial_idx env_uncertainty optimal_behavior expected_payoffs tau fixated stop
    )

    # Initialize model. 
    model = ABM(LearningAgent, scheduler = Schedulers.fastest;
                properties = params)

    # Initialize environment.
    model.optimal_behavior = sample(1:nbehaviors)
    model.expected_payoffs = repeat([low_payoff], nbehaviors)
    model.expected_payoffs[model.optimal_behavior] = high_payoff

    
    for ii in collect(1:numagents)

        if rand(model.rng) < init_social_learner_prevalence
            social_learner = true
        else
            social_learner = false
        end

        add_agent!(
            LearningAgent(;
                id = ii, 
                behavior = 0,  # Doesn't matter, will be selected at random at init.
                ledger = zeros(Float64, nbehaviors),
                behavior_count = zeros(Int64, nbehaviors),
                # social_learner = sample([true, false]),
                social_learner = social_learner,
                prev_social_learner = social_learner
            ), 
            model
        )

    end

    return model
end


"""
Learner agents do a behavior, perhaps are a social learner, accumulate payoffs,
and track expected payoffs.
"""
@with_kw mutable struct LearningAgent <: AbstractAgent
    
    # Constant factors and parameters.
    id::Int

    # Behavior represented by int that indexes payoffs to probabilistically
    # generate payoff.
    behavior::Int
    social_learner::Bool
    prev_social_learner::Bool

    # Payoffs. Need a step-specific payoff due to asynchrony--we don't want
    # some agents' payoffs to be higher just because they performed a behavior
    # before another.
    step_payoff::Float64 = 0.0
    net_payoff::Float64 = 0.0
    # Hold the "previous" net payoff to track net payoffs at final time step.
    prev_net_payoff::Float64 = 0.0

    # The ledger keeps track of individually-learned payoffs in each 
    # behavior-environment pair. Behaviors are rep'd by Int, which indexes
    # the ledger to look up previous payoffs and behavior counts.
    ledger::Array{Float64}
    behavior_count::Array{Int64}
end


"""
Generate a payoff that will be distributed to an agent performing the 
behavior with prob proportional to payoff for the chosen payoff.
"""
function add_step_payoff!(focal_agent::LearningAgent, model)

    # The focal_agent's behavior pays off probabilistically.
    if rand(model.rng) < model.expected_payoffs[focal_agent.behavior]
        payoff = 1.0
    else
        payoff = 0.0
    end

    # The step payoff is stored until all agents have asynchronously 
    # gotten their step payoff. Payoffs are added to net_payoffs and to the
    # agent's ledger in model_step!.
    focal_agent.step_payoff = payoff
end


function softmax(payoffs::AbstractVector, tau::Float64)

    exponentiated = exp.(payoffs ./ tau)
    denom = sum(exponentiated)

    return exponentiated ./ denom
end


function select_behavior!(focal_agent, model)
    
    weights = Weights(softmax(focal_agent.ledger, model.tau))
    focal_agent.behavior = sample(1:model.nbehaviors, weights)

    return nothing
end


"""
"""
function agent_step!(focal_agent::LearningAgent, 
                     model::ABM) 

    select_behavior!(focal_agent, model)
    add_step_payoff!(focal_agent, model)

end


"""

"""
function model_step!(model)

    for (idx, agent) in enumerate(collect(allagents(model)))
        prevledg = 0.0
     
        # Accumulate, record, and reset step payoff values.
        agent.net_payoff += agent.step_payoff
        agent.prev_net_payoff = copy(agent.net_payoff)
        
        # Update ledger and behavior counts.
        prevledg = agent.ledger[agent.behavior]
        agent.behavior_count[agent.behavior] += 1

        # Increment ledger amount from current behavior via exponential averaging.
        updated_ledger_amt = prevledg + (
                (agent.step_payoff - prevledg) / 
                agent.behavior_count[agent.behavior]
            )
        agent.ledger[agent.behavior] = updated_ledger_amt

        # Reset step payoff for the next time step.
        agent.step_payoff = 0.0

    end

    # If the model has gone steps_per_round time steps since the last model
    # update, evolve.
    if model.tick % model.steps_per_round == 0

        evolve!(model)

        if (model.env_uncertainty ≠ 0.0) && (rand(model.rng) < model.env_uncertainty)
            model.optimal_behavior = sample(filter(b -> b ≠ model.optimal_behavior,
                                            1:model.nbehaviors))

            model.expected_payoffs = repeat([model.low_payoff], model.nbehaviors)
            model.expected_payoffs[model.optimal_behavior] = model.high_payoff
        end

        for agent in allagents(model)

            # Reset net payoff and step payoffs.
            agent.net_payoff = 0.0
            agent.step_payoff = 0.0
        end

        # Next two conditionals detect when to stop for the 
        # stop-after-one-fixated-generation stopping function.
        model.stop = model.fixated

        # This one first detects whether the model is fixated, which will
        # be read above after one final generation runs.
        n_sl = sum(a.social_learner for a in allagents(model))
        model.fixated = (n_sl == 0.0) || (n_sl == nagents(model))
    end

    model.tick += 1
end


"""
Agents in the model 'evolve', which means they (1) produce offspring asexually 
with frequency proportional to relative payoffs---offspring inherit parent's
learning strategy with mutation; (2) die off.
"""
function evolve!(model::ABM)

    agents_coll = collect(allagents(model))

    parents = select_parents(model)
    parents_social_learner_trait = map(parent -> parent.social_learner, parents)

    for (idx, social_learner) in enumerate(parents_social_learner_trait)

        child = agents_coll[idx]
        child.prev_social_learner = copy(child.social_learner)
        child.social_learner = social_learner

        if child.social_learner
            agentsvec = collect(allagents(model)) 
            teachers = sample(agentsvec, model.nteachers, replace=false)

            teacher_idx = argmax(map(t -> t.net_payoff, teachers))
            teacher = teachers[teacher_idx]
            
            child.ledger = copy(teacher.ledger)
            # and the count of observations of each behavior is reset to 1.
            child.behavior_count = repeat([1], model.nbehaviors)

        else
            # If child is not a social learner, ledger and counts are totally reset.
            child.ledger = zeros(Float64, model.nbehaviors)
            child.behavior_count = zeros(Int64, model.nbehaviors)
        end
    end

end


"""
Sample N agents to be parents with replacement, weighted by net payoffs.
"""
function select_parents(model::ABM)

    all_net_payoffs = map(a -> a.net_payoff, allagents(model))

    N = nagents(model)
    
    parent_idxs = sample(
        collect(1:N), Weights(all_net_payoffs), N; replace=true
    )

    ret = collect(allagents(model))[parent_idxs]

    return ret

end
