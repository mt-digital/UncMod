using DrWatson
quickactivate("..")

using CategoricalArrays
using Chain

using DataFrames
using Gadfly, Compose, LaTeXStrings
using Glob
using Libz
using JLD2

import StatsBase: sample

# Not used here, but convenient for making notebooks.
import Cairo, Fontconfig

using Statistics

include("expected_homogenous_payoff.jl")


PROJECT_THEME = Theme(
    major_label_font="CMU Serif",minor_label_font="CMU Serif", 
    point_size=5.5pt, major_label_font_size = 18pt, 
    minor_label_font_size = 18pt, key_title_font_size=18pt, 
    line_width = 3.5pt, key_label_font_size=14pt
)

function N_sensitivity_results(yvars = 
                                [:mean_social_learner, :mean_prev_net_payoff, 
                                 :step]; 
                                Ns = ["50", "200", "1000"],
                                figuredir = "papers/UncMod/Figures", 
                                nbehaviorsvec=[2, 4, 10], 
                                ) 
    default_nfiles = 10

    for N in Ns
        for yvar in yvars
            # Currently have to manually separate files from N_sensitivity dir
            # from server into N=$N directories locally.
            # TODO Add code to check if these are available and create and automatically
            # separate if they are. This is done for compat with main_SL_result.
            datadir = "data/nagents_sensitivity/nagents=$N"

            if N == "1000"
                nfiles = 100
            else
                nfiles = default_nfiles
            end

            println("Using $nfiles files for N=$N")

            main_SL_result(yvar; figuredir = "$figuredir/nagents=$N", 
                           datadir, nfiles)
        end
    end
end


function nteachers_sensitivity_results(yvars = 
                                [:mean_social_learner, :mean_prev_net_payoff, 
                                 :step], 
                                nteachers_vals = ["2", "10", "20"];
                                figuredir = "papers/UncMod/Figures", 
                                nbehaviorsvec=[2, 4, 10], 
                                nfiles = 100)  # New parallel runs easily do 100 trials per file
    for nteachers in nteachers_vals
        for yvar in yvars
            # Currently have to manually separate files from N_sensitivity dir
            # from server into N=$N directories locally.
            # TODO Add code to check if these are available and create and automatically
            # separate if they are. This is done for compat with main_SL_result.
            # datadir = "data/nteachers_sensitivity/nteachers=$nteachers"
            datadir = "data/nteachers_sensitivity/nteachers$nteachers"
            main_SL_result(yvar; figuredir = "$figuredir/nteachers$nteachers", 
                           datadir, nfiles)
        end
    end
end

function tau_sensitivity_results(yvars = 
                                [:mean_social_learner, :mean_prev_net_payoff, 
                                 :step], 
                                taus = ["0.01", "1.0"];
                                figuredir = "papers/UncMod/Figures", 
                                nbehaviorsvec=[2, 4, 10], 
                                nfiles = 10)  # New parallel runs easily do 100 trials per file
    for tau in taus
        for yvar in yvars
            # Currently have to manually separate files from tau_sensitivity dir
            # from server into tau$tau directories locally.
            # TODO Add code to check if these are available and create and automatically
            # separate if they are. This is done for compat with main_SL_result.
            datadir = "data/tau_sensitivity/$tau"
            main_SL_result(yvar; figuredir = "$figuredir/sensitivity_tau=$tau", 
                           datadir, nfiles)
        end
    end
end


function main_SL_result(yvar = :mean_social_learner; 
                        figuredir = "papers/UncMod/Figures", 
                        nbehaviorsvec=[2, 4, 10], 
                        datadir = "data/develop",
                        nfiles = 100)  # Assumes 10 per trial, so 1000 trials.

    for nbehaviors in nbehaviorsvec
        # df = make_endtime_results_df("data/develop", nbehaviors, yvar)
        # Don't know why but this outperforms the above to build 
        # averaged dataframe at final time step.
        df = load_random_df(datadir, nbehaviors, nfiles)
        cdf = aggregate_final_timestep(df, yvar)
        plot_over_u_sigmoids(cdf, nbehaviors, yvar; figuredir = figuredir)
    end
end


function make_joined_from_file(model_outputs_file::String, ensemble_offset = nothing)
    outputs = load(model_outputs_file)

    adf, mdf = map(k -> outputs[k], ["adf", "mdf"])

    joined = innerjoin(adf, mdf, on = [:ensemble, :step]);

    if !isnothing(ensemble_offset)
        joined.ensemble .+= ensemble_offset
    end

    return joined
end


function make_endtime_results_df(model_outputs_dir::String, nbehaviors::Int, 
                                 yvar::Symbol)

    if isdir(model_outputs_dir)

        filepaths = glob("$model_outputs_dir/*nbehaviors=[$nbehaviors*")

        joined = vcat(
            [make_joined_from_file(f) for f in filepaths]...
        )
        
        return aggregate_final_timestep(joined, yvar)

    else

        error("$model_outputs_dir must be a directory but is not")
    end
end


function make_endtime_results_df(model_outputs_file::String, yvar::Symbol)

    joined = make_joined_from_file(model_outputs_file)

    return aggregate_final_timestep(joined, yvar)
end


function make_endtime_results_df(model_outputs_files::Vector{String}, 
                                 yvar::Symbol)

    joined = vcat(
        [make_joined_from_file(f, yvar) for f in model_outputs_files]...
    )
    
    return aggregate_final_timestep(joined, yvar)
end


function aggregate_final_timestep(joined_df::DataFrame, yvar::Symbol; 
                                  socdf = false, socdf_lifespan = 1, 
                                  generations = 1000)
    if socdf
        # Hack to deal with social learning dataframe problems.
        filter!(r -> r.step == socdf_lifespan * generations - 1, joined_df)
    end

    # Groupby ensemble, find maximum time step in each ensemble.
    gb = groupby(joined_df, :ensemble)

    cb = combine(gb, :step => maximum => :step)

    # Match ensemble and step, left join w combined on left,
    # so only rows from last time step remain.
    endstepdf = leftjoin(cb, joined_df; on = [:ensemble, :step])

    groupbydf = groupby(endstepdf, 
                        [:env_uncertainty, :steps_per_round, :low_payoff]);

    cdf = combine(groupbydf, yvar => mean => yvar)

    cdf.steps_per_round = categorical(cdf.steps_per_round)

    return cdf
end

using Colors
logocolors = Colors.JULIA_LOGO_COLORS
SEED_COLORS = [logocolors.purple, colorant"deepskyblue", 
               colorant"forestgreen", colorant"pink"] 
function gen_colors(n)
  cs = distinguishable_colors(n,
      SEED_COLORS, # seed colors
      lchoices=Float64[58, 45, 72.5, 90],     # lightness choices
      transform=c -> deuteranopic(c, 0.1),    # color transform
      cchoices=Float64[20,40],                # chroma choices
      hchoices=[75,51,35,120,180,210,270,310] # hue choices
  )

  convert(Vector{Color}, cs)
end

function gen_two_colors(n)
  cs = distinguishable_colors(n,
      [SEED_COLORS[1], SEED_COLORS[4]], # seed colors
      # [colorant"#FE4365", colorant"#eca25c"],
      lchoices=Float64[58, 45],     # lightness choices
      transform=c -> deuteranopic(c, 0.1),    # color transform
      cchoices=Float64[20,40],                # chroma choices
      hchoices=[75,51] # hue choices
  )

  convert(Vector{Color}, cs)
end


function plot_over_u_sigmoids(final_agg_df, nbehaviors, 
                                       yvar=:mean_social_learner; 
                                       low_payoffs=[0.1, 0.45, 0.8],
                                       figuredir=".")
    df = final_agg_df

    if yvar ∈ [:mean_prev_net_payoff, :step]
        if nbehaviors ∈ [2, 4]
            filter!(r -> r.steps_per_round ∈ [1, 8], df)
        else
            filter!(r -> r.steps_per_round ∈ [1, 20], df)
        end
    end

    if yvar == :mean_prev_net_payoff
        nfiles = 100
        socdf = load_random_df("data/sl_expected", nbehaviors, nfiles)
        if nbehaviors ∈ [2, 4]
            filter!(r -> r.steps_per_round ∈ [1, 8], socdf)
        else
            filter!(r -> r.steps_per_round ∈ [1, 20], socdf)
        end
        aggsocdf = aggregate_final_timestep(socdf, yvar; socdf = true)
        this_aggsocdf = aggsocdf[aggsocdf.low_payoff .== low_payoff, :]
    end

    for low_payoff in low_payoffs 

        thisdf = df[df.low_payoff .== low_payoff, :]

        if low_payoff == 0.8
            xlabel = "Env. variability, <i>u</i>"
        else
            xlabel = ""
        end

        if yvar == :mean_social_learner
            yticks = 0:.5:1
        end

        if yvar != :mean_prev_net_payoff

            if yvar == :step
                for r in eachrow(thisdf)
                    r[yvar] /= convert(Float64, r.steps_per_round)
                end
                ticloc = 20
                yticks = 0:ticloc:(ticloc * ceil(maximum(thisdf[!, yvar]) / ticloc))
            end
            colorgenfn = yvar == :step ? gen_two_colors : gen_colors 
            p = plot(thisdf, x=:env_uncertainty, y=yvar, 
                     color = :steps_per_round, Geom.line, Geom.point,
                     Guide.xlabel(""),
                     Guide.ylabel(""), 
                     Guide.yticks(ticks=yticks),
                     Scale.color_discrete(colorgenfn),
                     Guide.colorkey(title="<i>L</i>", pos=[.865w,-0.225h]),
                     PROJECT_THEME)
        else
            for r in eachrow(thisdf)
                r.mean_prev_net_payoff /= convert(Float64, r.steps_per_round)
            end

            # Prepare lines for expected individual learner payoffs, ⟨π_I⟩.
            indiv_file = "expected_individual.jld2"
            if !isfile(indiv_file)
                println("Expected individual payoffs not found, generating now...")
                # Calculates expected payoff for all parameter combos & saves.
                all_expected_individual_payoffs();
            end
            # indiv_df = load(indiv_file)["df"]
            indiv_df = load(indiv_file)["ret_df"]
            if nbehaviors ∈ [2, 4]
                spr = [1, 8]
            else
                spr = [1, 20]
            end
            indiv_df = filter(r -> (r.low_payoff == low_payoff) && 
                                   (r.nbehaviors == nbehaviors) &&
                                   (r.steps_per_round ∈ spr), 
                              indiv_df)
            
            for r in eachrow(indiv_df)
                r.mean_prev_net_payoff /= r.steps_per_round
            end

            expected_individual_intercepts = 
                sort(indiv_df, :steps_per_round).mean_prev_net_payoff

            # Prepare lines for expected social learner payoffs, ⟨πₛ⟩.
            # nfiles = 100 
            # df = load_random_df("data/sl_expected", nbehaviors, 10)
            # this_aggsocdf = aggregate_final_timestep(df, yvar)

            # println("Loading...")
            # this_aggsocdf = load(soc_file)["aggdf"]

            # this_aggsocdf = make_endtime_results_df("data/sl_expected", nbehaviors,
            filter!(r -> (r.low_payoff == low_payoff) &&
                         (r.steps_per_round ∈ spr),
                    this_aggsocdf)

            for r in eachrow(this_aggsocdf)
                r.mean_prev_net_payoff /= convert(Float64, r.steps_per_round)
            end

            yticks = 0.0:0.2:1
            p = plot(
                     layer(thisdf, x=:env_uncertainty, y=yvar, 
                           color = :steps_per_round, Geom.line, Geom.point),
                     yintercept = expected_individual_intercepts,
                     Geom.hline(; 
                                color=[SEED_COLORS[1], 
                                       SEED_COLORS[4]], 
                                style=:ldash, size=2.5pt),
                     layer(this_aggsocdf, x=:env_uncertainty, y=yvar, Geom.line,
                           Geom.point,
                           color = :steps_per_round, 
                           style(point_shapes=[diamond],
                                 point_size=4.5pt,
                                 line_width=2.5pt, 
                                 line_style=[:dashdot])), 
                     Guide.xlabel(""),
                     Guide.ylabel(""), 
                     Guide.yticks(ticks=yticks),
                     Scale.color_discrete(gen_two_colors),
                     Guide.colorkey(title="<i>L</i>", pos=[.865w,-0.225h]),
                     PROJECT_THEME)
        end
        
        draw(
             PDF(joinpath(
                 figuredir, 
                 "$(yvar)_over_u_lowpayoff=$(low_payoff)_nbehaviors=$(nbehaviors).pdf"), 
                 4.25inch, 3inch), 
            p
        )
    end
end



function plot_timeseries_selection(datadir, low_payoff, nbehaviors, 
                                   env_uncertainty, steps_per_round, ntimeseries,
                                   yvar = :mean_social_learner,
                                   series_per_file = 10)

    # Default 50 series for each parameter setting in each file.
    nfiles = Int(ceil(ntimeseries / series_per_file))
    df = load_random_df(datadir, nbehaviors, nfiles)
    df.ensemble = string.(df.ensemble)
    
    select_cond = (df.env_uncertainty .== env_uncertainty) .&
                  (df.steps_per_round .== steps_per_round) .&
                  (df.low_payoff .== low_payoff)

    df = df[select_cond, :]
    
    nsteps = length(unique(df.step))
    nensembles = length(unique(df.ensemble))
    ensemble_replace_indexes = 
        string.(vcat([repeat([i], nsteps) for i in 1:nensembles]...))

    df.ensemble = ensemble_replace_indexes
    
    # Make plot.
    pilow = low_payoff
    B = nbehaviors
    u = env_uncertainty
    L = steps_per_round

    title = 
"""π<sub> low</sub> = $pilow; B = $B; u = $env_uncertainty; L = $L
"""

    plot(df, x=:step, y=:mean_social_learner, color=:ensemble, 
         Geom.line(), Guide.title(title))
end


function load_random_df(datadir::String, nbehaviors::Int, nfiles::Int)

    globstring = "$datadir/*nbehaviors=[$nbehaviors*"
    println("Loading data from files matching $globstring")

    globlist = glob(globstring)
    filepaths = sample(globlist, nfiles)

    dfs = []
    ensemble_offset = 0

    for f in filepaths
        tempdf = make_joined_from_file(f, ensemble_offset)
        push!(dfs, tempdf)
        ensemble_offset = maximum(tempdf.ensemble)
    end

    return vcat(dfs...)
end


function load_random_df(datadir::String, nfiles::Int, 
                        nbehaviors::Vector{Int}=[2,4,10])

    return vcat([load_random_df(datadir, B, nfiles) for B in nbehaviors]...)
end


"Calculate tally of how many non-fixated trials there are compared to total
trials for each dataframe passed in from vector. Assumed there is one dataframe
for each behavior, so each dataframe has exactly one unique value in its
nbehaviors column."
function calculate_pct_fixation(df::DataFrame)
    
    # Initialize output tally of fixated and total trials.
    fixdf = DataFrame(
        :B => Int[], :NotFixated => Int[], :TotalTrials => Int[],
        :PctNotFixated => Real[]
    )

    # Iterate over given dataframes, assumed one for each behavior.
    # for df in behavior_dfs

    # print(last(df, 20))

    by_ensemble = groupby(df, [:nbehaviors, :ensemble])
    # println(by_ensemble)

    cb = combine(by_ensemble, :step => maximum => :step)

    endstepdf = leftjoin(cb, df; on = [:nbehaviors, :ensemble, :step])

    endstepdf.nonfix = (endstepdf.mean_social_learner .> 0.0) .&
                       (endstepdf.mean_social_learner .< 1.0)

    res = combine(groupby(endstepdf, :nbehaviors), 
                     :nonfix => sum => :nonfix)

    res.nonfixpct = res.nonfix / (nrow(endstepdf) / length(res.nbehaviors))

    return nrow(endstepdf), res
end


function calculate_pct_fixation(datadir::String, nfiles::Int, 
                                nbehaviors::Vector{Int}=[2,4,10])

    df = load_random_df(datadir, nfiles, nbehaviors)

    return df, calculate_pct_fixation(df)
end
