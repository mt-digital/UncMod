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
                                    mutation_magnitude = 0.1,  # σₘ in paper.
                                    # learnparams_mutating = [:homophily, :exploration, :soclearnfreq],
                                    # learnparams_mutating = [:soclearnfreq],
                                    base_reliabilities = [0.5, 0.5],
                                    reliability_variance = 0.1,
                                    steps_per_round = 10,
                                    ntoreprodie = 10,
                                    nteachers = 10,
                                    regen_reliabilities = false,
                                    init_soclearnfreq = 0.0,
                                    τ_init = 0.01,
                                    dτ = 9.999e-5,
                                    # payoff_learning_bias = false,
                                    high_payoff = nothing,
                                    low_payoff = nothing,
                                    nbehaviors = nothing,
                                    trial_idx = nothing,
                                    annealing = true,
                                    vertical = true,
                                    # Says how much to reduce the ledger when 
                                    # passed between generations.
                                    model_parameters...)
    
    if isnothing(nbehaviors)
        nbehaviors = length(base_reliabilities)
    else
        base_reliabilities = [low_payoff for _ in 1:nbehaviors]
        base_reliabilities[1] = high_payoff
    end

    tick = 1

    # XXX not sure why/if this is necessary.
    # steps_per_round += 1

    # Build full dictionary of model parameters and mutation distribution.
    params = merge(

        Dict(model_parameters), 
        
        Dict(:mutation_distro => Normal(0.0, mutation_magnitude)),

        @dict steps_per_round ntoreprodie tick base_reliabilities reliability_variance  nbehaviors nteachers τ_init regen_reliabilities low_payoff high_payoff trial_idx annealing vertical
    )
    
    # Initialize model. 
    model = ABM(LearningAgent, scheduler = Schedulers.fastest;
                properties = params)

    function add_soclearn_agent!(idx::Int64)
        # For now initialize behaviors randomly. These can be modified after
        # initialization as needed for different experiments.
        add_agent!(LearningAgent(
                       id = idx,
                       behavior = sample(1:nbehaviors),
                       reliabilities = 
                        draw_reliabilities(base_reliabilities, reliability_variance),
                       ledger = zeros(Float64, nbehaviors),
                       behavior_count = zeros(Int64, nbehaviors),
                       soclearnfreq = init_soclearnfreq,
                       τ = τ_init,
                       dτ = dτ
                   ), 
                   model)
    end
    
    for ii in 1:nagents
        add_soclearn_agent!(ii)
    end

    return model
end


"""

"""
@with_kw mutable struct LearningAgent <: AbstractAgent
    
    # Constant factors and parameters.
    id::Int

    # Behavior represented by int that indexes reliabilities to probabilistically
    # generate payoff.
    behavior::Int64
    reliabilities::Array{Float64}

    # Learning parameters.
    soclearnfreq::Float64 
    vertical_squeeze::Float64 = 0.1

    # Softmax temperature
    τ::Float64 
    # Softmax annealing subtractive change 
    dτ::Float64

    # Payoffs. Need a step-specific payoff due to asynchrony--we don't want
    # some agents' payoffs to be higher just because they performed a behavior
    # before another.
    prev_net_payoff::Float64 = 0.0
    step_payoff::Float64 = 0.0
    net_payoff::Float64 = 0.0

    # The ledger keeps track of individually-learned payoffs in each 
    # behavior-environment pair. Behaviors are rep'd by Int, which indexes
    # the ledger to look up previous payoffs and behavior counts.
    ledger::Array{Float64}
    behavior_count::Array{Int64}

    age::Int64 = 0

    uuid::UUID = uuid4()
    parent::Union{UUID, Nothing} = nothing
end






"""
Generate a payoff that will be distributed to an agent performing the 
behavior with prob proportional to reliability for the chosen payoff.
"""
function generate_payoff!(focal_agent::LearningAgent)  #behavior_idx::Int64, reliabilities::Array{Float64})

    # println(focal_agent.reliabilities)
    if rand() < focal_agent.reliabilities[focal_agent.behavior]
        payoff = 1.0
    else
        payoff = 0.0
    end

    focal_agent.step_payoff = payoff

    return payoff
end


# Convert the desired reliability mean and variance to Beta dist parameters.
function μσ²_to_αβ(μ, σ²)

    α::Float64 = (μ^2) * (((1 - μ) / σ²) - (1/μ))
    β::Float64 = α * ((1 / μ) - 1)

    return α, β
end


function draw_reliabilities(base_reliabilities, reliability_variance)

    # Transform base reliabilities and reliability variance into Beta dist params.
    params = map(base_rel -> μσ²_to_αβ(base_rel, reliability_variance),
                        base_reliabilities)

    return map(
        ((α, β),) -> rand(Beta(α, β)),
        params
    )
end


function learn_behavior(focal_agent::LearningAgent, 
                        model::ABM,
                        teachers = nothing)

    # If no teachers are provided this Asocial learning
    if isnothing(teachers)
        if !(sum(focal_agent.ledger) == 0.0)
            weights = Weights(softmax(focal_agent.ledger, focal_agent.τ))
            behavior = sample(1:model.nbehaviors, weights)
        else 
            behavior = focal_agent.behavior
        end
    else
        weights = Weights(map(a -> a.net_payoff, teachers))
        behavior = sample(teachers, weights).behavior
    end

    return behavior
end


function softmax(payoffs::AbstractVector, τ::Float64)

    exponentiated = exp.(payoffs ./ τ)
    denom = sum(exponentiated)

    return exponentiated ./ denom
end


function select_behavior!(focal_agent, model)
    
    # First, determine whether learning is individual or social for this step.
    focal_soclearnfreq = focal_agent.soclearnfreq    

    if (rand() < focal_soclearnfreq) 
        teachers = sample(
            filter(a -> a ≠ focal_agent, collect(allagents(model))), 
            model.nteachers
        )
        behavior = learn_behavior(focal_agent, model, teachers)
    else
        behavior = learn_behavior(focal_agent, model)
    end

    focal_agent.behavior = behavior

    return behavior
end


"""
"""
function agent_step!(focal_agent::LearningAgent, 
                     model::ABM)

    select_behavior!(focal_agent, model)
    generate_payoff!(focal_agent)

end



"""
"""
function model_step!(model)

    # println(model.step)
    for agent in allagents(model)
        # Accumulate, record, and reset step payoff values.
        agent.prev_net_payoff = agent.net_payoff
        agent.net_payoff += agent.step_payoff
        
        # Update ledger and behavior counts.
        prevledg = agent.ledger[agent.behavior]
        agent.behavior_count[agent.behavior] += 1
        # println(prevledg)
        updated_ledger_amt = prevledg + (
                (agent.step_payoff - prevledg) / 
                agent.behavior_count[agent.behavior]
                # Float64(agent.behavior_count[agent.behavior])
            )
        # println(updated_ledger_amt)
        # println(agent.behavior_count)
        agent.ledger[agent.behavior] = updated_ledger_amt

        # Reset payoffs for the next time step.
        agent.step_payoff = 0.0
        # Softmax annealing.
        if model.annealing
            agent.τ -= agent.dτ
        end
        
        # if model.regen_reliabilities
            # agent.reliabilities = draw_reliabilities(model.base_reliabilities,
            #                                          model.reliability_variance)
        # end
    end

    # If the model has gone steps_per_round time steps since the last model
    # update, evolve the three social learning traits.
    if model.tick % model.steps_per_round == 0

        evolve!(model)

        for agent in allagents(model)

            agent.prev_net_payoff = 0.0
            agent.net_payoff = 0.0
            agent.behavior = sample(1:model.nbehaviors)

            # Reset softmax temperature to re-start in-round annealing.
            agent.τ = model.τ_init

            if model.regen_reliabilities
                agent.reliabilities = draw_reliabilities(
                    model.base_reliabilities, model.reliability_variance
                )
            end
        end

    end

    model.tick += 1
end


function select_reproducers(model::ABM)
    all_net_payoffs = map(a -> a.net_payoff, allagents(model))

    N = nagents(model)
    select_idxs = sample(
        1:N, Weights(all_net_payoffs), N; replace=true
    )
    
    ret = collect(allagents(model))[select_idxs]
    return ret
end


function select_to_die(model, reproducers)
    agents = allagents(model)

    N = nagents(model)
    all_idxs = 1:N
    repro_ids = map(a -> a.id, reproducers)
    # XXX we are not allowing reproducing agents to possibly die. Problem? XXX
    available_idxs = filter(idx -> !(idx ∈ repro_ids), all_idxs)
    agent_ages = map(id -> model[id].age, available_idxs)
    select_idxs = sample(available_idxs, 
                         Weights(agent_ages), 
                         model.properties[:ntoreprodie],
                         replace = false)

    return filter(a -> a.id ∈ select_idxs, collect(allagents(model)))
end


function repro_with_mutations!(model, parent, child)
    
    # Overwrite dead agent's information either with unique information or
    # properties of parent as appropriate.
    child.uuid = uuid4()
    child.age = 0

    # Setting dead agent's fields with relevant repro agent's, no mutation yet.
    child.parent = parent.uuid

    if model.vertical
        transmit_vertical!(parent, child)
    else
        child.ledger = zeros(Float64, model.nbehaviors)
        child.behavior_count = zeros(Int64, model.nbehaviors)
    end
    
    # Social learning frequency and vertical squeeze amount are both inherited
    # with mutation.
    if model.vertical
        mutparams = [:soclearnfreq, :vertical_squeeze]
    else
        mutparams = [:soclearnfreq]
    end

    for mutparam in mutparams
        mutdistro = model.mutation_distro
        newparamval = getproperty(parent, mutparam) + rand(mutdistro)

        # Learning parameters are limited to [0.0, 1.0].
        if newparamval > 1.0
            newparamval = 1.0
        elseif newparamval < 0.0
            newparamval = 0.0
        end
        
        # Set child agent to have mutated param value.
        setproperty!(
            child, mutparam, newparamval
        )
    end
end


function transmit_vertical!(parent, child)

    parent_mean = mean.(parent.ledger)

    child.ledger = 
        parent_mean + 
        (parent.vertical_squeeze*(parent_mean .- parent.ledger))

    child.behavior_count = 
        Integer.(
            floor.(parent.behavior_count .* parent.vertical_squeeze)
        )
end


"""
Agents in the model 'evolve', which means they (1) produce offspring asexually 
with frequency proportional to relative payoffs---offspring inherit parent's
learning strategy with mutation; (2) die off.
"""
function evolve!(model::ABM)

    reproducers = select_reproducers(model)
    terminals = collect(allagents(model))

    for (idx, repro_agent) in enumerate(reproducers)
        repro_with_mutations!(model, repro_agent, terminals[idx])
    end
end
