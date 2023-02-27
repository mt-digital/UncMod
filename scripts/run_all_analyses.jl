using DrWatson; quickactivate("."); 
using RCall
include("analysis.jl")


# Main analysis figures in order of appearance.
main_SL_result(:mean_social_learner)
# Figure 3 of social learning ceiling; R code called from Julia using RCall.jl.
R"""
source("scripts/plot.R")
sl_ceiling_plot()
"""
main_SL_result(:mean_prev_net_payoff)
main_SL_result(:step)

# Figure 5A---note we cherry-picked outcomes to demonstrate how social or 
# asocial learning can sometimes counter-intuitively evolve despite homogenous 
# population payoffs being greater for asocial or social learning, respectively, 
# counter-intuitively occur due to finite population sizes and stochastic 
# environmental variability. Due to this stochasticity, these functions may
# need to be run multiple times to get a simulation where this reversal occurs.
u = 0.2; pilow = 0.1; B = 2; L = 4;
sim_agent_df, sim_model_df = make_payoffs_timeseries(u, pilow, B, L)
plot_payoff_timeseries(sim_agent_df, sim_model_df, u, pilow, B, L)

# Figure 5B.
u = 0.5; pilow = 0.1; B = 4; L = 1;
sim_agent_df, sim_model_df = make_payoffs_timeseries(u, pilow, B, L)
plot_payoff_timeseries(sim_agent_df, sim_model_df, u, pilow, B, L)

N_sensitivity_results()
nteachers_sensitivity_results()
# We reported inverse temperature, i.e. "greediness", beta in paper, but used
# softmax temperature, i.e. exploration paramater, in code, hence "tau" here.
tau_sensitivity_results()
