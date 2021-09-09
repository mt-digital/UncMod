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
# using Random


# Agents may be in one of two groups, call them A and B.
@enum Group GroupA GroupB

# Agents may perform one of two behaviors, call them A and B.
@enum Behavior BehaviorA BehaviorB

# Agents may adopt one of three social learning strategies.
@enum SocialLearningStrategy Rand Conformist Success

# Two basic types of learning strategies.
# @enum LearningStrategy Individual Social

# Environments may be in one of two payoff states.
@enum PayoffState Low High

@enum EnvironmentLabel EnvA EnvB

"Agents may be in one of two (or a number of) environments."
struct Environment
    label::EnvironmentLabel
    state::PayoffState
    behavior::Behavior  # The behavior that gets payoff_multiplier*(w + b) payoff.
    uncertainty::Float64
    payoff_multiplier::Float64
end


# mutable struct IndividualLearningLedger
#     "Combinations of {BEHAVIOR}{ENVIRONMENT}"
#     AA::Float64
#     AB::Float64
#     BA::Float64
#     BB::Float64
# end

# function reset_ledger!(ledger::IndividualLearningLedger)
#     ledger.AA = 0.0; ledger.AB = 0.0; ledger.BA = 0.0; ledger.BB = 0.0;
# end

"Agent social learning strategies have three heritable components"
struct LearningStrategy
    soc_learn_freq::Float64
    sl_strategy::SocialLearningStrategy
    parochialism::Float64
end


IndividualLearningLedger = Dict{Tuple{Behavior, EnvironmentLabel}, Float64}

"""
"""
mutable struct LearningAgent <: AbstractAgent
    # Constant factors and parameters.
    id::Int
    group::Group

    # Location may be one of two environments.
    location::Environment

    # Learned environmental strategy.
    behavior::Behavior
    
    # Evolved factors and parameters.
    learning_strategy::LearningStrategy

    # Payoffs. Need a step-specific payoff due to asynchrony--we don't want
    # some agents' payoffs to be higher just because they performed a behavior
    # before another.
    prev_payoff::Float64
    step_payoff::Float64
    net_payoff::Float64

    # The ledger keeps track of individually-learned payoffs in each 
    # behavior-environment pair.
    ledger::IndividualLearningLedger
end







"""
Step the environment forward in time to possibly change its state
with probability depending on initialization parameters.
"""
function environment_step!(environment::Environment)
    # If random uniform draw less than (1 - ambiguity), change payoff state.
    if environment.uncertainty < rand()
        if environment.state == Low
            environment.state = High
        else
            environment.state = Low
        end
    end
end


"""
Calculate payoff to focal_agent operating in the g
"""
function getpayoff!(focal_agent::LearningAgent, 
                    environment::Environment,
                    model::ABM)

    # Calculate payoff given agent strategy and environment.
    if focal_agent.behavior == environment.behavior
        payoff = payoff_multiplier(environment, model) * (model.W + model.B) 
    else
        payoff = payoff_multiplier(environment, model) * (model.W - model.B) 
    end
    
    # Increase net payoffs for focal_agent.
    focal_agent.step_payoff = payoff
end


function payoff_multiplier(environment)
    # return draw from normal distribution with params spec'd to environment state.
end



"""
Arguments:
    group_env_corr (Float64): Amount initial behaviors & identities correlate
        with environment.
"""
function init_agent(agent_idx::Int64, group::Group, group_env_corr::Float64)

    # TODO Create and set variables to construct new LearningAgent.
    LearningAgent(agent_idx, group ... )
    
end



"""
"""
function uncertainty_learning_model(numagents=6, ; model_parameters...)
    
    # Initialize model. 
    model = ABM(LearningAgent, scheduler = Schedulers.fastest;
                properties = params)
    
    for ii in 1:numagents
        if ii < (numagents / 2)
            add_agent!(init_agent(ii, GroupA, group_env_corr), model)
        else
            add_agent!(init_agent(ii, GroupB, gropu_env_corr), model)
        end
    end
end


"""
"""
function agent_step!(focal_agent::LearningAgent, 
                     environment::Environment,
                     model::ABM)

    # First, determine whether learning is individual or social for this step.
    learning_strategy::LearningStrategy = Individual
    # if random uniform draw > freq_indiv_learn then set to Social
    if learning_strategy == Social
        # select teacher based on parochialism and social learning strategy.
        teachers = filter(
            other_agent -> other_agent.id != focal_agent.id,
            collect(allagents(model))
        )
    end

    getsteppayoff!(focal_agent, environment, model)
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

    
    if rand() < agent.soc_learn_freq
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
# function terminate(model, s)

# end


"""
"""
function single_trial(; kwargs...)
    model = model_fun_name(; kwargs...)

    agent_data, model_data = run!(model, agent_step!, model_step!, terminate;
             # adata = agentdata_columns,
             # mdata = modeldata_columns
             )

    return agent_data, model_data
end
