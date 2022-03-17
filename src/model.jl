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

"""
function uncertainty_learning_model(; 
                                    nagents = 100,  
                                    # minority_frac = 0.5, 
                                    mutation_magnitude = 0.05,  # σₘ in paper.
                                    # learnparams_mutating = [:homophily, :exploration, :soclearnfreq],
                                    # learnparams_mutating = [:soclearnfreq],
                                    steps_per_round = 5,
                                    nteachers = 5,
                                    # regen_payoffs = false,
                                    init_soclearnfreq = 0.0,
                                    τ_init = 0.01,
                                    # dτ = 9.999e-5,
                                    dτ = 0.0,
                                    # payoff_learning_bias = false,
                                    high_payoff = nothing,
                                    low_payoff = nothing,
                                    nbehaviors = nothing,
                                    trial_idx = nothing,
                                    # annealing = true,
                                    # vertical = true,
                                    env_uncertainty = 0.0,
                                    # Says how much to reduce the ledger when 
                                    # passed between generations.
                                    model_parameters...)
    

    tick = 1

    # Build full dictionary of model parameters and mutation distribution.
    params = merge(

        Dict(model_parameters),  
        
        # Dict(:mutation_distro => Normal(0.0, mutation_magnitude),
         Dict(:optimal_behavior => 1),

        @dict steps_per_round tick low_payoff high_payoff nbehaviors nteachers τ_init trial_idx env_uncertainty mutation_prob
    )
    
    # Initialize model. 
    model = ABM(LearningAgent, scheduler = Schedulers.fastest;
                properties = params)

    function add_soclearn_agent!(idx::Int64)
        # For now initialize behaviors randomly. These can be modified after
        # initialization as needed for different experiments.
        add_agent!(LearningAgent(id = idx, τ = τ_init, dτ = dτ), model)
                       # behavior = sample(1:nbehaviors),
                       # ledger = zeros(Float64, nbehaviors),
                       # behavior_count = zeros(Int64, nbehaviors),
                       # social_learner = sample([true, false]),
    end
    
    for ii in 1:nagents
        # add_soclearn_agent!(ii)
        add_agent!(LearningAgent(id = ii, τ = τ_init, dτ = dτ), model)
    end

    return model
end


"""

"""
@with_kw mutable struct LearningAgent <: AbstractAgent
    
    # Constant factors and parameters.
    id::Int

    # Behavior represented by int that indexes payoffs to probabilistically
    # generate payoff.
    behavior::Int = sample(1:nbehaviors)
    social_learner::Bool = sample([false, true])

    # Softmax temperature
    τ::Float64 
    # Softmax annealing subtractive change 
    # dτ::Float64

    # Payoffs. Need a step-specific payoff due to asynchrony--we don't want
    # some agents' payoffs to be higher just because they performed a behavior
    # before another.
    prev_net_payoff::Float64 = 0.0
    step_payoff::Float64 = 0.0
    net_payoff::Float64 = 0.0

    # The ledger keeps track of individually-learned payoffs in each 
    # behavior-environment pair. Behaviors are rep'd by Int, which indexes
    # the ledger to look up previous payoffs and behavior counts.
    ledger::Array{Float64} = zeros(Float64, nbehaviors)
    behavior_count::Array{Int64} = zeros(Int64, nbehaviors)

    age::Int64 = 0

    uuid::UUID = uuid4()
    parent::Union{UUID, Nothing} = nothing
end


"""
Generate a payoff that will be distributed to an agent performing the 
behavior with prob proportional to payoff for the chosen payoff.
"""
function add_step_payoff!(focal_agent::LearningAgent)

    if rand() < model.expected_payoffs[focal_agent.behavior]
        payoff = 1.0
    else
        payoff = 0.0
    end

    # Here the step payoff is stored until all agents have asynchronously 
    # gotten their step payoff. Payoffs are be added to net_payoffs and to the
    # agent's ledger in model_step!.
    focal_agent.step_payoff = payoff

    return nothing
end


# function learn_behavior(focal_agent::LearningAgent, 
#                         model::ABM,
#                         teachers = nothing)

#     # If no teachers are provided this Asocial learning
#     if isnothing(teachers)
#         if !(sum(focal_agent.ledger) == 0.0)
#             weights = Weights(softmax(focal_agent.ledger, focal_agent.τ))
#             behavior = sample(1:model.nbehaviors, weights)
#         else 
#             behavior = focal_agent.behavior
#         end
#     else
#         weights = Weights(map(a -> a.net_payoff, teachers))
#         behavior = sample(teachers, weights).behavior
#     end

#     return behavior
# end


function softmax(payoffs::AbstractVector, τ::Float64)

    exponentiated = exp.(payoffs ./ τ)
    denom = sum(exponentiated)

    return exponentiated ./ denom
end


function select_behavior!(focal_agent, model)
    
    weights = Weights(softmax(focal_agent.ledger, focal_agent.τ))
    focal_agent.behavior = sample(1:model.nbehaviors, weights)

    return nothing
end


"""
"""
function agent_step!(focal_agent::LearningAgent, 
                     model::ABM) 

    select_behavior!(focal_agent, model)
    add_step_payoff!(focal_agent)

end



"""
"""
function model_step!(model)

    for agent in allagents(model)
     
        # Accumulate, record, and reset step payoff values.
        agent.prev_net_payoff = agent.net_payoff
        agent.net_payoff += agent.step_payoff
        
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

        # Softmax annealing.
        if model.annealing
            agent.τ -= agent.dτ
        end
        
    end

    # If the model has gone steps_per_round time steps since the last model
    # update, evolve the three social learning traits.
    if model.tick % model.steps_per_round == 0

        evolve!(model)

        if rand() < model.env_uncertainty
            model.optimal_behavior = sample(1:model.nbehaviors)
        end

        for agent in allagents(model)

            agent.prev_net_payoff = 0.0
            agent.net_payoff = 0.0

            # TODO probably not actually necessary–check
            agent.behavior = sample(1:model.nbehaviors)

            # Reset softmax temperature to re-start in-round annealing.
            agent.τ = model.τ_init
        end

    end

    model.tick += 1
end


"""
Sample N agents to be parents with replacement, weighted by net payoffs.
"""
function select_parents(model::ABM)

    all_net_payoffs = map(a -> a.net_payoff, allagents(model))

    N = nagents(model)

    parent_idxs = sample(
        1:N, Weights(all_net_payoffs), N; replace=true
    )
    
    # return collect(allagents(model))[parent_idxs]
    return filter(a -> a.id ∈ parent_idxs, allagents(model))
end


function repro_learn_with_mutations!(model, parent, child)
    
    # Overwrite dead agent's information either with unique information or
    # properties of parent as appropriate.
    child.uuid = uuid4()
    child.age = 0

    # Setting dead agent's fields with relevant repro agent's, no mutation yet.
    child.parent = parent.uuid

    # Social learning frequency and vertical squeeze amount are both inherited
    # with mutation.
    if (model.mutation_prob > 0) && (rand() < model.mutation_prob)
        child.social_learner = ~parent.social_learner
    else
        child.social_learner = parent.social_learner
    end

    if child.social_learner
        # Child takes the ledger from parent...
        child.ledger = parent.ledger
        # and the count of observations of each behavior is reset to 1.
        child.behavior_count = repeat([1], model.nbehaviors)
    else
        # If child is not a social learner, ledger and counts are totally reset.
        child.ledger = zeros(Float64, model.nbehaviors)
        child.behavior_count = zeros(Int64, model.nbehaviors)
    end
end


"""
Agents in the model 'evolve', which means they (1) produce offspring asexually 
with frequency proportional to relative payoffs---offspring inherit parent's
learning strategy with mutation; (2) die off.
"""
function evolve!(model::ABM)

    for (idx, parent) in enumerate(select_parents(model))

        repro_learn_with_mutations!(
            model, repro_agent, collect(allagents(model))
        )

    end
end
