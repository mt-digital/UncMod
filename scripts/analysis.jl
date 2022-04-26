using DrWatson
quickactivate("..")

using DataFrames
using Gadfly, Compose
using Glob
using Libz
using JLD2

# Not used here, but convenient for making notebooks.
import Cairo, Fontconfig

using Statistics


PROJECT_THEME = Theme(
    major_label_font="CMU Serif",minor_label_font="CMU Serif", 
    point_size=3.5pt, major_label_font_size = 18pt, 
    minor_label_font_size = 14pt, key_title_font_size=14pt, 
    line_width = 2pt, key_label_font_size=12pt
)


function main_SL_result(figure_dir = "papers/UncMod/Figures", 
                        nbehaviorsvec=[2, 4, 10]; datadir = "data/develop")

    for nbehaviors in nbehaviorsvec
        df = make_endtime_results_df("data/develop", nbehaviors)
        plot_soclearn_over_u_sigmoids(df, nbehaviors; figure_dir = figure_dir)
    end
end


function make_joined_from_file(model_outputs_file::String)

    outputs = load(model_outputs_file)

    adf, mdf = map(k -> outputs[k], ["adf", "mdf"])

    joined = innerjoin(adf, mdf, on = [:ensemble, :step]);
    
    max_step = maximum(joined.step)

    joined = joined[
        joined.step .== max_step, 
        [:countmap_behavior, :mean_social_learner, :env_uncertainty, 
         :low_payoff, :nbehaviors, :steps_per_round, :optimal_behavior]
    ]

    return joined
end


function make_endtime_results_df(model_outputs_dir::String, nbehaviors::Int)

    if isdir(model_outputs_dir)

        filepaths = glob("$model_outputs_dir/*nbehaviors=[$nbehaviors*")

        joined = vcat(
            [make_joined_from_file(f) for f in filepaths]...
        )
        
        return aggregate_final_timestep(joined)

    else

        error("$model_outputs_dir must be a directory but is not")
    end

end


function make_endtime_results_df(model_outputs_file::String)

    joined = make_joined_from_file(model_outputs_file)

    return aggregate_final_timestep(joined)
end


function make_endtime_results_df(model_outputs_files::Vector{String})

    joined = vcat(
        [make_joined_from_file(f) for f in model_outputs_files]...
    )
    
    return aggregate_final_timestep(joined)
end


function aggregate_final_timestep(joined_df::DataFrame)

    groupbydf = groupby(joined_df, 
                        [:env_uncertainty, :steps_per_round, :low_payoff]);

    cdf = combine(groupbydf, :mean_social_learner => mean)

    cdf.steps_per_round = string.(cdf.steps_per_round)

    return cdf
end

using Colors
logocolors = Colors.JULIA_LOGO_COLORS
function gen_colors(n)
  cs = distinguishable_colors(n,
      [logocolors.purple, colorant"deepskyblue", 
       colorant"forestgreen", colorant"pink"], # seed colors
      # [colorant"#FE4365", colorant"#eca25c"],
      lchoices=Float64[58, 45, 72.5, 90],     # lightness choices
      transform=c -> deuteranopic(c, 0.1),    # color transform
      cchoices=Float64[20,40],                # chroma choices
      hchoices=[75,51,35,120,180,210,270,310] # hue choices
  )

  convert(Vector{Color}, cs)
end


function plot_soclearn_over_u_sigmoids(final_agg_df, nbehaviors; 
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

        p = plot(thisdf, x=:env_uncertainty, y=:mean_social_learner_mean, 
                 color = :steps_per_round, Geom.line, Geom.point,
                 Theme(line_width=1.5pt), 
                 Guide.xlabel(""),
                 Guide.ylabel(""), 
                 Guide.yticks(ticks=0.0:0.5:1.0),
                 Scale.color_discrete(gen_colors),
                 Guide.colorkey(title="<i>L</i>", pos=[.865w,-0.225h]),
                 PROJECT_THEME)

        draw(
             PDF(joinpath(
                 figure_dir, 
                 "SL_over_u_lowpayoff=$(low_payoff)_nbehaviors=$(nbehaviors).pdf"), 
                 4.25inch, 3inch), 
            p
        )
    end
end

