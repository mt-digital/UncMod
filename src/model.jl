##
# Model of the evolution of social learning under group membership and uncertain
# environmental conditions.
#
# Author: Matthew A. Turner
# Date: 2021-09-06
#


using Agents
using Distributions
using DrWatson
using Parameters
using StatsBase
using UUIDs




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
    soclearnfreq::Float64 = 0.0
    exploration::Float64 = 0.01

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

    if rand() < focal_agent.reliabilities[focal_agent.behavior]
        payoff = 1.0
    else
        payoff = 0.0
    end

    focal_agent.step_payoff = payoff

    return payoff
end


# """
# Arguments:
#     model: Contains initial behaviors & identities correlate with environment.
# """
# function init_learning_strategy(group, model)

#     params = model.properties

#     if haskey(params, :nteachers) 
#         learning_strategy = LearningStrategy(nteachers = model.properties[:nteachers])

#     elseif haskey(params, :initlearningstrategies)
#         learning_strategy = params[:initlearningstrategies][group]

#     else
#         learning_strategy = LearningStrategy()

#     end
    
#     return learning_strategy
# end

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

"""
"""
function uncertainty_learning_model(; 
                                    nagents = 100, 
                                    minority_frac = 0.5, 
                                    mutation_magnitude = 0.1,  # σₘ in paper.
                                    # learnparams_mutating = [:homophily, :exploration, :soclearnfreq],
                                    # learnparams_mutating = [:soclearnfreq],
                                    base_reliabilities = [0.5, 0.5],
                                    reliability_variance = 0.1,
                                    steps_per_round = 10,
                                    ntoreprodie = 10,
                                    nteachers = 10,
                                    # payoff_learning_bias = false,
                                    model_parameters...)
    
    nbehaviors = length(base_reliabilities)

    tick = 1

    params = merge(
        Dict(model_parameters), 
        
        Dict(:mutation_distro => Normal(0.0, mutation_magnitude)),

        @dict steps_per_round ntoreprodie tick base_reliabilities reliability_variance minority_frac nbehaviors nteachers
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
                       behavior_count = zeros(Int64, nbehaviors)
                   ), 
                   model)
    end
    
    for ii in 1:nagents
        add_soclearn_agent!(ii)
    end

    return model
end


function learn_behavior(focal_agent::LearningAgent, 
                        model::ABM,
                        teachers = nothing)

    if isnothing(teachers)
        # Asocial learning.
        if !(sum(focal_agent.ledger) == 0.0)
            behavior = findmax(focal_agent.ledger)[2]
        else 
            behavior = focal_agent.behavior
        end
    else
        # Social learning via conformist transmission, i.e., adopt most common behavior.
        # n_using_behaviors = countmap([t.behavior for t in teachers])
        # # Index of the maximum is the second argument returned from findmax.
        # behavior = findmax(n_using_behaviors)[2]
        
        # Added 10/12/2021 for push for results before AABA abstract deadline
        weights = Weights(map(a -> a.net_payoff, teachers))
        behavior = sample(teachers, weights).behavior
    end

    # Whatever the learning result, override what was learned and select at 
    # random with probability equal to agent's exploration value in 
    # learning strategy.
    if rand() < focal_agent.exploration
        behavior = sample(1:model.nbehaviors)
    end

    return behavior
end


function select_behavior!(focal_agent, model)
    
    # First, determine whether learning is individual or social for this step.
    focal_soclearnfreq = focal_agent.soclearnfreq    

    if (focal_soclearnfreq == 1.0) || (rand() < focal_soclearnfreq) 
        teachers = sample(
            filter(a -> a ≠ focal_agent, collect(allagents(model))), 
            model.nteachers
        )
        behavior = learn_behavior(focal_agent, model, teachers)
    else
        # Select behavior based on individual learning with probability ϵ.
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

    for agent in allagents(model)
        # Accumulate, record, and reset step payoff values.
        agent.prev_net_payoff = agent.net_payoff
        agent.net_payoff += agent.step_payoff
        agent.behavior_count[agent.behavior] += 1
        agent.ledger[agent.behavior] += (
            agent.ledger[agent.behavior] + 
            agent.step_payoff) / Float64(agent.behavior_count[agent.behavior])
        agent.step_payoff = 0.0
    end

    # If the model has gone steps_per_round time steps since the last model
    # update, evolve the three social learning traits.
    if model.tick % model.steps_per_round == 0
        foreach(a -> a.age += 1, allagents(model))
        # reproducers = select_reproducers(model)
        # terminals = select_to_die(model)

        evolve!(model)  #, reproducers, terminals)

        for agent in allagents(model)
            agent.prev_net_payoff = 0.0
            agent.net_payoff = 0.0
            agent.ledger = zeros(Float64, model.nbehaviors)
            agent.behavior_count = zeros(Int64, model.nbehaviors)
            agent.behavior = sample(1:model.nbehaviors)
            agent.reliabilities = draw_reliabilities(model.base_reliabilities, model.reliability_variance)
        end
    end

    model.tick += 1
end

function select_reproducers(model::ABM)
    all_net_payoffs = map(a -> a.net_payoff, allagents(model))

    N = nagents(model)
    select_idxs = sample(
        1:N, Weights(all_net_payoffs), model.ntoreprodie; replace=false
    )

    ret = collect(allagents(model))[select_idxs]
    return ret
end

function select_to_die(model, reproducers)
    agents = allagents(model)

    N = nagents(model)
    all_idxs = 1:N
    repro_ids = map(a -> a.id, reproducers)
    available_idxs = filter(idx -> !(idx ∈ repro_ids), all_idxs)
    agent_ages = map(id -> model[id].age, available_idxs)
    select_idxs = sample(available_idxs, 
                         Weights(agent_ages), 
                         model.properties[:ntoreprodie],
                         replace = false)

    return filter(a -> a.id ∈ select_idxs, collect(allagents(model)))
end


function repro_with_mutations!(model, repro_agent, dead_agent)
    
    # Overwrite dead agent's information either with unique information or
    # properties of repro_agent as appropriate.
    dead_agent.uuid = uuid4()
    dead_agent.age = 0
    dead_agent.ledger = zeros(Float64, model.nbehaviors)
    dead_agent.behavior_count = zeros(Int64, model.nbehaviors)

    # Setting dead agent's fields with relevant repro agent's, no mutation yet.
    dead_agent.parent = repro_agent.uuid
    dead_agent.behavior = repro_agent.behavior
    for field in [:behavior]
        setproperty!(dead_agent, field, getproperty(repro_agent, field))
    end

    # Mutate one of the learning strategy parameters selected at random if more
    # than one is being allowed to evolve.
    # mutparam = sample(model.properties[:learnparams_mutating])
    mutparam = :soclearnfreq  # For now set only param that mutates; see above.
    mutdistro = model.mutation_distro
    newparamval = getproperty(repro_agent, mutparam) + rand(mutdistro)

    # All learning parameter values are probabilities, thus limited to [0.0, 1.0].
    if newparamval > 1.0
        newparamval = 1.0
    elseif newparamval < 0.0
        newparamval = 0.0
    end

    # dead_agent.learning_strategy = repro_agent.learning_strategy
    setproperty!(
        dead_agent, mutparam, newparamval
    )

end


"""
Agents in the model 'evolve', which means they (1) produce offspring asexually 
with frequency proportional to relative payoffs---offspring inherit parent's
learning strategy with mutation; (2) die off.
"""
function evolve!(model::ABM)

    reproducers = select_reproducers(model)
    terminals = select_to_die(model, reproducers)

    for (idx, repro_agent) in enumerate(reproducers)
        repro_with_mutations!(model, repro_agent, terminals[idx])
    end
end
