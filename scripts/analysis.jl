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


function main_SL_result(yvar = :mean_social_learner, 
                        figure_dir = "papers/UncMod/Figures", 
                        nbehaviorsvec=[2, 4, 10]; 
                        datadir = "data/develop")

    for nbehaviors in nbehaviorsvec
        df = make_endtime_results_df("data/develop", nbehaviors, yvar)
        plot_over_u_sigmoids(df, nbehaviors, yvar; 
                             figure_dir = figure_dir)
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

    max_step = maximum(joined_df.step)
    joined_df = joined_df[joined_df.step .== max_step, :]

    groupbydf = groupby(joined_df, 
                        [:env_uncertainty, :steps_per_round, :low_payoff]);

    cdf = combine(groupbydf, yvar => mean => yvar)

    cdf.steps_per_round = string.(cdf.steps_per_round)

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
                                       figure_dir=".")
    df = final_agg_df

    for low_payoff in low_payoffs 

        thisdf = df[df.low_payoff .== low_payoff, :]

        if low_payoff == 0.8
            xlabel = "Env. variability, <i>u</i>"
        else
            xlabel = ""
        end

        if yvar != :mean_prev_net_payoff
            p = plot(thisdf, x=:env_uncertainty, y=yvar, 
                     color = :steps_per_round, Geom.line, Geom.point,
                     Theme(line_width=1.5pt), 
                     Guide.xlabel(""),
                     Guide.ylabel(""), 
                     Guide.yticks(ticks=0.0:0.5:1.0),
                     Scale.color_discrete(gen_colors),
                     Guide.colorkey(title="<i>L</i>", pos=[.865w,-0.225h]),
                     PROJECT_THEME)
        else
            df = load("expected_individual.jld2")["ret_df"]
            df = filter(r -> (r.low_payoff == low_payoff) && 
                             (r.nbehaviors == nbehaviors), 
                        df)

            expected_individual_intercepts = 
                sort(df, :steps_per_round)[:mean_prev_net_payoff]
            
            p = plot(thisdf, x=:env_uncertainty, y=yvar, 
                     color = :steps_per_round, Geom.line, Geom.point,
                     yintercept = expected_individual_intercepts,
                     Geom.hline(; color=SEED_COLORS, style=:dot),
                     Theme(line_width=1.5pt), 
                     Guide.xlabel(""),
                     Guide.ylabel(""), 
                     Guide.yticks(ticks=0.0:0.5:1.0),
                     Scale.color_discrete(gen_colors),
                     Guide.colorkey(title="<i>L</i>", pos=[.865w,-0.225h]),
                     PROJECT_THEME)
        end
        
        draw(
             PDF(joinpath(
                 figure_dir, 
                 "$(yvar)_over_u_lowpayoff=$(low_payoff)_nbehaviors=$(nbehaviors).pdf"), 
                 4.25inch, 3inch), 
            p
        )
    end
end



function plot_timeseries_selection(datadir, low_payoff, nbehaviors, 
                                   env_uncertainty, steps_per_round, ntimeseries,
                                   yvar = :mean_social_learner,
                                   series_per_file = 50)

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
        categorical(vcat([repeat([i], nsteps) for i in 1:nensembles]...))

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


function load_random_df(datadir, nbehaviors, nfiles)

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
