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


# Agents may be in one of two groups, call them 1 and 2.
@enum Group Group1 Group2


# Agents may perform one of two behaviors, call them 1 and 2.
@enum Behavior Behavior1 Behavior2


# Each behavior may be in one of two payoff states.
@enum PayoffState Low High


"""
The BehaviorPayoffStructure is provided by the 'Environment', which is not
directly represented in this model, but is useful to talk about as the 
thing that yields payoffs with a specified BehaviorPayoffStructure. 
The Environment
"""
struct BehaviorPayoffStructure
    high_payoff::Float64
    low_state_frac::Float64 # cᵢ, which reduces so the Low payoff to cᵢπᵢ. 
    reliability::Float64 # How likely behavior results in High payoff, ρᵢ.
end


"""
Generate a payoff that will be distributed to an agent performing the 
behavior according to 
"""
function generate_payoff(payoff_structure::BehaviorPayoffStructure)
    
    # With probability equal to "uncertainty" the BehaviorPayoff will be in a
    # Low state.
    if rand() < payoff_structure.reliability
        payoff = payoff_structure.high_payoff
    else
        payoff = payoff_structure.high_payoff * payoff_structure.low_state_frac
    end

    return payoff
end


"Agent social learning strategies have three heritable components"
@with_kw mutable struct LearningStrategy

    # Frequency of social learning versus asocial learning (trial and error).
    soclearnfreq::Float64 = rand() 
    
    # Actually "greediness" value ϵ, adapted for our context. This one can
    # be updated through social learning.
    exploration::Float64 = rand()
    
    # Homophily of 0 indicates teachers chosen randomly; 1 indcates 
    # an in-group member is always chosen. Again, could be evolved or
    # influenced.
    homophily::Float64 = rand()

    # How many others we learn from could be an important parameter.
    nteachers::Float64 = 10
end



"""

"""
@with_kw mutable struct LearningAgent <: AbstractAgent
    
    # Constant factors and parameters.
    id::Int
    group::Group

    # Learned environmental strategy.
    behavior::Behavior
    
    # Learning strategy components that may be set constant, learned, or
    # evolved; see LearningStrategy struct above for definition.
    learning_strategy::LearningStrategy = LearningStrategy()

    # Payoffs. Need a step-specific payoff due to asynchrony--we don't want
    # some agents' payoffs to be higher just because they performed a behavior
    # before another.
    prev_net_payoff::Float64 = 0.0
    step_payoff::Float64 = 0.0
    net_payoff::Float64 = 0.0

    # The ledger keeps track of individually-learned payoffs in each 
    # behavior-environment pair.
    ledger::Dict{Behavior, Float64} = Dict(Behavior1 => 0.0, Behavior2 => 0.0)
    behavior_count::Dict{Behavior, Int64} = Dict(Behavior1 => 0, Behavior2 => 0)

    age::Int64 = 0

    uuid::UUID = uuid4()
    parent::Union{UUID,Nothing} = nothing
end


"""
Arguments:
    group: Groups are possibly correlated with behavior
    model: Contains initial behaviors & identities correlate with environment.
"""
function init_learning_strategy(group, model)

    params = model.properties

    if haskey(params, :nteachers) 
        learning_strategy = LearningStrategy(nteachers = model.properties[:nteachers])

    elseif haskey(params, :initlearningstrategies)
        learning_strategy = params[:initlearningstrategies][group]

    else
        learning_strategy = LearningStrategy()

    end
    
    return learning_strategy
end


"""
"""
function uncertainty_learning_model(; 
                                    nagents = 100,
                                    minority_frac = 0.5, 
                                    environment = Dict(
                                        Behavior1 => 
                                            BehaviorPayoffStructure(1.0, 0.5, 0.5),
                                        Behavior2 => 
                                            BehaviorPayoffStructure(1.0, 0.5, 0.5),
                                    ),
                                    mutation_magnitude = 0.05,  # σₘ in paper.
                                    learnparams_mutating = [:soclearnfreq],
                                    # payoff_learning_bias = false,
                                    model_parameters...)
    
    params = merge(
        Dict(model_parameters), 
        Dict(:minority_frac => minority_frac, 
             :environment => environment,
             :learnparams_mutating => learnparams_mutating,
             :mutation_distro => Normal(0.0, mutation_magnitude))
    )
    
    # Initialize model. 
    model = ABM(LearningAgent, scheduler = Schedulers.fastest;
                properties = params)

    function add_soclearn_agent!(idx::Int64, group::Group)
        # For now initialize behaviors randomly. These can be modified after
        # initialization as needed for different experiments.
        add_agent!(LearningAgent(
                       id = idx,
                       group = group, 
                       behavior = rand(instances(Behavior)),
                       learning_strategy = init_learning_strategy(group, model) 
                   ), 
                   model)
    end
    
    ngroup1 = round(nagents * minority_frac)
    for ii in 1:nagents
        if ii <= ngroup1
            add_soclearn_agent!(ii, Group1)
        else
            add_soclearn_agent!(ii, Group2)
        end
    end

    # Deal with group-behavior correlations for Group1/Behavior1 correlation.
    if haskey(params, :R0_1) 

        n1 = round((params[:R0_1] * ngroup1) + 0.01)  # add extra for rounding up from X.5.
        g1agents = filter(a -> a.group == Group1, collect(allagents(model)))
        b1_idxs = sample(1:length(g1agents), Int(n1), replace=false)

        for (agent_idx, agent) in enumerate(g1agents)
            if agent_idx ∈ b1_idxs
                model[agent.id].behavior = Behavior1
            else
                model[agent.id].behavior = Behavior2
            end
        end
    end

    # Deal with group-behavior correlations for Group2/Behavior2 correlation.
    if haskey(params, :R0_2)

        n2 = round(
            (params[:R0_2] * (nagents - ngroup1)) + 0.01  # add extra for rounding up from X.5.
        )
        g2agents = filter(a -> a.group == Group2, collect(allagents(model)))
        b2_idxs = sample(1:length(g2agents), Int(n2), replace=false)

        for (agent_idx, agent) in enumerate(g2agents)
            if agent_idx ∈ b2_idxs
                model[agent.id].behavior = Behavior2
            else
                model[agent.id].behavior = Behavior1
            end
        end
    end

    return model
end


function learn_behavior(focal_agent::LearningAgent, 
                        teachers = nothing)

    if isnothing(teachers)
        # Asocial learning.
        if !(focal_agent.ledger[Behavior1] + focal_agent.ledger[Behavior2] == 0.0)
            behavior = findmax(focal_agent.ledger)[2]
        else 
            behavior = focal_agent.behavior
        end
    else
        # Social learning via conformist transmission, i.e., adopt most common behavior.
        n_using1 = count(behavior -> behavior == Behavior1,
                         map(teacher -> teacher.behavior, teachers))

        if n_using1 > (length(teachers) / 2.0)
            behavior = Behavior1
        elseif n_using1 < (length(teachers) / 2.0)
            behavior = Behavior2 
        else 
            behavior = rand([Behavior1, Behavior2])
        end
    end

    # Whatever the learning result, override what was learned and select at 
    # random with probability equal to agent's exploration value in 
    # learning strategy.
    if rand() < focal_agent.learning_strategy.exploration
        behavior = sample([Behavior1, Behavior2], 1)[1]
    end

    return behavior
end


function select_behavior!(focal_agent, model)
    
    # First, determine whether learning is individual or social for this step.
    focal_soclearnfreq = focal_agent.learning_strategy.soclearnfreq    

    if (focal_soclearnfreq == 1.0) || (rand() < focal_soclearnfreq) 
        # Select teacher based on parochialism and social learning strategy.
        teachers = select_teachers(focal_agent, model)
        behavior = learn_behavior(focal_agent, teachers)
    else
        # Select behavior based on individual learning with probability ϵ
        behavior = learn_behavior(focal_agent)
    end

    focal_agent.behavior = behavior

    return behavior
end


"""
"""
function agent_step!(focal_agent::LearningAgent, 
                     model::ABM)

    behavior = select_behavior!(focal_agent, model)

    # focal_agent.step_payoff = generate_payoff(
    focal_agent.step_payoff = generate_payoff(
        model.properties[:environment][behavior]
    )

    # focal_agent.net_payoff += step_payoff
end


function select_teachers(focal_agent::LearningAgent, model::ABM)

    in_group = focal_agent.group
    if in_group == Group1
        out_group = Group2
    else
        out_group = Group1
    end

    teachers = LearningAgent[]
    possible_teachers = filter(agent -> agent != focal_agent, collect(allagents(model)))

    nteachers = focal_agent.learning_strategy.nteachers

    for _ in 1:nteachers

        homophily = focal_agent.learning_strategy.homophily

        if (homophily == 1.0) || 
           (rand() < ((1 + homophily) / 2.0))

            teacher_group = in_group
        else
            teacher_group = out_group
        end

        group_possible_teachers = filter(
            agent -> (agent.group == teacher_group),
            possible_teachers
        )

        if length(group_possible_teachers) == 0
            group_possible_teachers = possible_teachers
        end
        
        teacheridx = rand(1:length(group_possible_teachers))
        teacher = group_possible_teachers[teacheridx]
        push!(teachers, teacher)

        deleteat!(possible_teachers, 
                  findfirst(t -> t == teacher, possible_teachers))
        
    end

    return teachers
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
    if model.step_counter % model.steps_per_round == 0
        foreach(a -> a.age += 1, allagents(model))
        reproducers = select_reproducers(model)
        terminals = select_to_die(model)

        evolve!(model, reproducers, terminals)

        for agent in allagents(model)
            agent.prev_net_payoff = 0.0
            agent.net_payoff = 0.0
        end
    end
end

function select_reproducers(model::ABM)
    all_net_payoffs = map(a -> a.net_payoff, allagents(model))

    N = nagents(model)
    select_idxs = sample(1:nagents(model), Weights(all_net_payoffs), 
                         model.properties[:ntoreprodie])

    return collect(allagents(model))[select_idxs]
end

function select_todie(model)
    agent_ages = map(a -> a.age, allagents(model))

    N = nagents(model)
    select_idxs = sample(1:nagents(model), 
                         Weights(agent_ages), 
                         model.properties[:ntoreprodie])

    return collect(allagents(model))[select_idxs]
end


function repro_with_mutations!(model, repro_agent, dead_agent)
    
    # Overwrite dead agent's information either with unique information or
    # properties of repro_agent as appropriate.
    dead_agent.uuid = uuid4()
    dead_agent.age = 0
    dead_agent.ledger = Dict(Behavior1 => 0.0, Behavior2 => 0.0)
    dead_agent.behavior_count = Dict(Behavior1 => 0, Behavior2 => 0)

    # Setting dead agent's fields with relevant repro agent's, no mutation yet.
    dead_agent.parent = repro_agent.uuid
    for field in [:group, :behavior, :learning_strategy]
        setproperty!(dead_agent, field, getproperty(repro_agent, field))
    end

    # Mutate one of the learning strategy parameters selected at random if more
    # than one is being allowed to evolve.
    mutparam = sample(model.properties[:learnparams_mutating])
    mutdistro = model.properties[:mutation_distro]
    setproperty!(
        dead_agent.learning_strategy, mutparam, 
        getproperty(repro_agent.learning_strategy, mutparam) + rand(mutdistro)
    )

end


"""
Agents in the model 'evolve', which means they (1) produce offspring asexually 
with frequency proportional to relative payoffs---offspring inherit parent's
learning strategy with mutation; (2) die off.
"""
function evolve!(model::ABM)

    reproducers = select_reproducers(model)
    terminals = select_todie(model)

    for (idx, repro_agent) in enumerate(reproducers)
        repro_with_mutations!(model, repro_agent, terminals[idx])
    end
end
