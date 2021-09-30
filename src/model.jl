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
@with_kw struct LearningStrategy

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

    age::Int64 = 0
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
                                            BehaviorPayoffStructure(0.5, 2.0, 0.5),
                                        Behavior2 => 
                                            BehaviorPayoffStructure(0.25, 1.0, 0.25)
                                    ),
                                    # payoff_learning_bias = false,
                                    model_parameters...)
    
    params = merge(
        Dict(model_parameters), 
        Dict(:minority_frac => minority_frac),
        Dict(:environment => environment)
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
    
    for ii in 1:nagents
        if ii < (nagents * minority_frac / 2)
            add_soclearn_agent!(ii, Group1)
        else
            add_soclearn_agent!(ii, Group2)
        end
    end

    return model
end

function learn_behavior(focal_agent::LearningAgent, 
                        teachers = nothing)

    if isnothing(teachers)
        # Asocial learning.
        behavior = findmax(focal_agent.ledger)[2]
    else
        # Social learning via conformist transmission, i.e., adopt most common behavior.
        n_using1 = count(behavior -> behavior == Behavior1,
                         map(teacher -> teacher.behavior, teachers))

        if n_using1 > length(teachers) / 2.0
            behavior = Behavior1
        elseif n_using1 < length(teachers) / 2.0
            behavior = Behavior2 
        else 
            behavior = rand([Behavior1, Behavior2])
        end
    end

    # Whatever the learning result, override what was learned and select at 
    # random with probability equal to agent's exploration value in 
    # learning strategy.
    if rand() < focal_agent.learning_strategy.exploration
        behavior = rand([Behavior1, Behavior2])
    end

    return behavior
end


function select_behavior!(focal_agent, model)
    
    # First, determine whether learning is individual or social for this step.
    focal_soclearnfreq = focal_agent.learning_strategy.soclearnfreq    

    if (focal_soclearnfreq == 1.0) || (rand() < focal_soclearnfreq) 
        # Select teacher based on parochialism and social learning strategy.
        teachers = select_teachers(focal_agent, model)
        # println("Teachers:")
        # println(teachers)
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

    behavior = select_behavior!(focal_agent)

    # focal_agent.step_payoff = generate_payoff(
    step_payoff = generate_payoff(
        model.properties.payoff_structures[behavior]
    )

    focal_agent.net_payoff += step_payoff
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
        
        # println("Possible teachers:")
        # println(map(t -> t.id, possible_teachers))
        # println()
        # println("Teachers:")
        # println(map(t -> t.id, teachers))
        # println()


        # if model.properties.payoff_bias

        #     possible_teacher_payoffs = map(t -> t.prev_net_payoff,
        #                                    possible_teachers)

        #     push!(
        #         teachers, 
        #         sample(possible_teachers, Weights(map(t -> t.net_payoff)))
        #     )
        # end
    end

    return teachers
end



"""
"""
function model_step!(model)
    # On every time step after each agent calculates its payoff, 
    # accumulate payoffs to each agent. Need to wait until all agents have
    # calculated step-specific payoffs to add them to net payoff for purposes
    # of social learning teacher selection at each step.
    for a in allagents(model)
        
        # Agents learn individually or socially depending on strategy and payoffs.
        learn!(a, model)

        # Accumulate, record, and reset step payoff values.
        a.payoffs += a.step_payoff
        a.prev_step_payoff = a.step_payoff
        a.step_payoff = 0.0
    end

    # If the model has gone steps_per_round time steps since the last model
    # update, evolve the three social learning traits.
    if model.step_counter % model.steps_per_round == 0
        evolve!(model)
    end
end


"""
Agents learn individually or socially randomly, weighted by their individual
frequency of social learning. 
"""
function learn!(agent::LearningAgent, model::ABM)

    
    if (agent.soc_learn_freq == 1.0) || (rand() < agent.soc_learn_freq)
       # TODO 

    end
end


"""
Agents in the model 'evolve', which means they (1) produce offspring asexually 
with frequency proportional to relative payoffs---offspring inherit parent's
learning strategy with mutation; (2) die off.
"""
function evolve!(model::ABM)

    all_prev_agents = deepcopy(allagents(model))

    all_net_payoffs = map(a -> a.net_payoff, prev_agents)

    N = nagents(model)
    ids_to_reproduce = sample(1:N, all_net_payoffs, N)

    for (idx, child) in enumerate(allagents(model))

        parent = all_prev_agents[idx]

        child.learning_strategy = parent.learning_strategy
        if rand() < mutate_freq  
            # Randomly select one of the three learning strategy components
            # to perturb 
        end

        child_agent.group = parent.group
        child_agent.net_payoff = 0
    end
end

"""
Perhaps terminate when value p has stabilized for all agents? For now, though,
we can just run for a certain number of steps and I'll comment this out.
"""
function terminate(model, s)
    if model.properties.n_rounds_elapsed == model.n_rounds_max
        return true
    else
        return false
    end
end
