using DrWatson
quickactivate("..")

using CategoricalArrays

using DataFrames
using Gadfly, Compose, LaTeXStrings
using Glob
using Libz
using JLD2

import StatsBase: sample

# Not used here, but convenient for making notebooks.
import Cairo, Fontconfig

using Statistics



PROJECT_THEME = Theme(
    major_label_font="CMU Serif",minor_label_font="CMU Serif", 
    point_size=3.5pt, major_label_font_size = 18pt, 
    minor_label_font_size = 14pt, key_title_font_size=14pt, 
    line_width = 2pt, key_label_font_size=12pt
)


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


function aggregate_final_timestep(joined_df::DataFrame, yvar::Symbol)

    # Groupby ensemble, find maximum time step in each ensemble.
    gb = groupby(joined_df, :ensemble)
    cb = combine(gb, :step => maximum => :step)

    # Match ensemble and step, left join w combined on left,
    # so only rows from last time step remain.
    endstepdf = leftjoin(cb, joined_df; on = [:ensemble, :step])
    # println(first(endstepdf, 20))
    # max_step = maximum(joined_df.step)
    # joined_df = joined_df[joined_df.step .== max_step, :]

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
      # [colorant"#FE4365", colorant"#eca25c"],
      lchoices=Float64[58, 45, 72.5, 90],     # lightness choices
      transform=c -> deuteranopic(c, 0.1),    # color transform
      cchoices=Float64[20,40],                # chroma choices
      hchoices=[75,51,35,120,180,210,270,310] # hue choices
  )

  convert(Vector{Color}, cs)
end


function plot_over_u_sigmoids(final_agg_df, nbehaviors, 
                                       yvar=:mean_social_learner; 
                                       low_payoffs=[0.1, 0.45, 0.8],
                                       figuredir=".")
    df = final_agg_df

    for low_payoff in low_payoffs 

        thisdf = df[df.low_payoff .== low_payoff, :]

        if low_payoff == 0.8
            xlabel = "Env. variability, <i>u</i>"
        else
            xlabel = ""
        end
        if yvar == :mean_social_learner
            yticks = 0:.5:1
        elseif yvar == :mean_prev_net_payoff
            if nbehaviors == 10
                yticks = 0:2:20
            else
                yticks = 0:1:8
            end
        elseif yvar == :step
            yticks = 0:500:(500 * ceil(maximum(thisdf[!, yvar]) / 500))
        end

        if yvar != :mean_prev_net_payoff
            p = plot(thisdf, x=:env_uncertainty, y=yvar, 
                     color = :steps_per_round, Geom.line, Geom.point,
                     Theme(line_width=1.5pt), 
                     Guide.xlabel(""),
                     Guide.ylabel(""), 
                     Guide.yticks(ticks=yticks),
                     Scale.color_discrete(gen_colors),
                     Guide.colorkey(title="<i>L</i>", pos=[.865w,-0.225h]),
                     PROJECT_THEME)
        else
            indiv_file = "expected_individual.jld2"
            if !isfile(indiv_file)
                println("Expected individual payoffs not found, generating now...")
                include("expected_individual_payoff.jl")
                # Calculates expected payoff for all parameter combos & saves.
                all_expected_payoffs();
            end
            indiv_df = load(indiv_file)["ret_df"]
            indiv_df = filter(r -> (r.low_payoff == low_payoff) && 
                             (r.nbehaviors == nbehaviors), 
                        indiv_df)

            expected_individual_intercepts = 
                sort(indiv_df, :steps_per_round).mean_prev_net_payoff
            
            p = plot(thisdf, x=:env_uncertainty, y=yvar, 
                     color = :steps_per_round, Geom.line, Geom.point,
                     yintercept = expected_individual_intercepts,
                     Geom.hline(; color=SEED_COLORS, style=:dot),
                     Theme(line_width=1.5pt), 
                     Guide.xlabel(""),
                     Guide.ylabel(""), 
                     Guide.yticks(ticks=yticks),
                     Scale.color_discrete(gen_colors),
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

    println(length(df.ensemble))
    println(length(ensemble_replace_indexes))
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


function load_expected_social_dfs(nbehaviors::Int; datadir = "data/expected_social", 
                                    jld2_key = "expected_social_joined_df"
    )

    d = Dict{Int, DataFrame}()

    if nbehaviors == 10
        filepaths_10 = glob("$datadir/*nbehaviors=[[]10*") 
        dfs = Vector{DataFrame}()
        ensemble_offset = 0
        
        for f in filepaths_10

            tempdf = load(f)[jld2_key]
            tempdf.ensemble .+= ensemble_offset
            ensemble_offset = maximum(tempdf.ensemble)

            push!(dfs, tempdf)
        end

        df10 = vcat(dfs...) 
        
        return df10
        
    else
        dfs = Vector{DataFrame}()
        ensemble_offset = 0

        filepaths_2_4 = glob("$datadir/*nbehaviors=[[]2,4[]]*")
        for f in filepaths_2_4

            tempdf = load(f)[jld2_key]
            tempdf.ensemble .+= ensemble_offset
            ensemble_offset = maximum(tempdf.ensemble)

            push!(dfs, tempdf)
        end

        df_2_4 = vcat(dfs...)
        
        return df_2_4
    end
end
                            

function load_random_df(datadir::String, nbehaviors::Int, nfiles::Int)

    filepaths = sample(glob("$datadir/*nbehaviors=[$nbehaviors*"),
                       nfiles)

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
function calculate_pct_fixation(behavior_dfs::Vector{DataFrame})
    
    # Initialize output tally of fixated and total trials.
    fixdf = DataFrame(
        :B => Int[], :NotFixated => Int[], :TotalTrials => Int[],
        :PctNotFixated => Real[]
    )

    # Iterate over given dataframes, assumed one for each behavior.
    for df in behavior_dfs

        by_ensemble = groupby(df, :ensemble)
        cb = combine(by_ensemble, :step => maximum => :step)

        endstepdf = leftjoin(cb, df; on = [:ensemble, :step])

        nonfix = endstepdf[(endstepdf.mean_social_learner .> 0.0) .& 
                           (endstepdf.mean_social_learner .< 1.0), 
                           :]
        N_trials = nrow(endstepdf)
        N_nonfix = nrow(nonfix)
        push!(fixdf, [df[1, :nbehaviors], N_nonfix, N_trials, (N_nonfix / N_trials)])
    end

    # Want table to ordered by increasing B.
    return sort(fixdf, :B)
end


function calculate_pct_fixation(datadir::String, nfiles::Int, 
                                nbehaviors::Vector{Int}=[2,4,10])

    df = load_random_df(datadir, nfiles, nbehaviors)

    return df, calculate_pct_fixation(df)
end
