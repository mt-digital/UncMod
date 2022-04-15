using DrWatson
quickactivate("..")

using DataFrames
using Gadfly, Compose
using Libz
using JLD2

# Not used here, but convenient for making notebooks.
import Cairo, Fontconfig

using Statistics


PROJECT_THEME = Theme(
    point_size=3.5pt, major_label_font_size = 16pt, 
    minor_label_font_size = 14pt, key_title_font_size=13pt, 
    line_width = 2pt, key_label_font_size=12pt
)




function make_endtime_results_df(model_outputs_file)
    outputs = load(model_outputs_file)
    adf, mdf = map(k -> outputs[k], ["adf", "mdf"])

    res = innerjoin(adf, mdf, on = [:ensemble, :step]);

    endtimesdf = filter(r -> (r.step == 999) && (r.low_payoff == 0.25), res);
    # first(endtimesdf, 10)

    groupbydf = groupby(res, [:env_uncertainty, :steps_per_round, :low_payoff]);

    cdf = combine(groupbydf, :mean_social_learner => mean)
    cdf.steps_per_round = string.(cdf.steps_per_round);

    return cdf
end


function plot_soclearn_over_u_sigmoids(final_agg_df, nbehaviors; 
                                       low_payoffs=[0.1, 0.45, 0.8],
                                       figure_dir=".")
    df = final_agg_df
    


    for low_payoff in low_payoffs 
        thisdf = df[df.low_payoff .== low_payoff, :]

        p = plot(thisdf, x=:env_uncertainty, y=:mean_social_learner_mean, 
                 color = :steps_per_round, Geom.line, Geom.point,
                 Theme(line_width=1.5pt), 
                 Guide.xlabel("Env. var, u"), 
                 Guide.ylabel("Social learning"), 
                 Guide.yticks(ticks=0.0:0.5:1.0),
                 Guide.title("Low payoff = $low_payoff"),
                 # Guide.colorkey(pos=[0.045w, 0.3h], title="<i>M</i>"),
                 Guide.colorkey(title="<i>M</i>"),
                 PROJECT_THEME)

        draw(
             PDF(joinpath(figure_dir, "SL_over_u_lowpayoff=$(low_payoff)_nbehaviors=$(nbehaviors).pdf"), 4.25inch, 3inch), 
            p
        )
    end
end


