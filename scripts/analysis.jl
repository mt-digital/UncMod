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
    line_width = 3.5pt, key_label_font_size=14pt, grid_line_width = 3.5pt
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
                        opacity = 0.8,
                        figuredir = "papers/UncMod/Figures", 
                        nbehaviorsvec=[2, 4, 10], 
                        datadir = "data/develop",
                        nfiles = 100)  # Assumes 10 per trial, so 1000 trials.

    for nbehaviors in nbehaviorsvec
        # df = make_endtime_results_df("data/develop", nbehaviors, yvar)
        # Don't know why but this outperforms the above to build 
        # averaged dataframe at final time step.
        aggdf_file = "data/mainResult-yvar=$yvar-B=$nbehaviors.jld2"
        if isfile(aggdf_file)
            println("Loading aggregated data from file $aggdf_file")
            aggdf = load(aggdf_file)["aggdf"]
        else
            df = load_random_df(datadir, nbehaviors, nfiles)
            aggdf = aggregate_final_timestep(df, yvar)
            @save aggdf_file aggdf
        end

        plot_over_u_sigmoids(aggdf, nbehaviors, yvar; figuredir, nfiles, opacity)
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
                                  socdf = false, generations = 1000
    )

    if socdf
        # Hack to deal with social learning dataframe problems.
        # println("doing filter")
        # print(last(joined_df, 10))
        # print(unique(joined_df.step))
        # if joined_df.nbehaviors[1] == 10
        #     filter!(r -> r.step == r.steps_per_round * 5000, joined_df)
        # else
        filter!(r -> r.step == (r.steps_per_round * generations), joined_df)
        # end

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

  return map(c -> RGBA(c, 0.4), cs)
end

function gen_colors_transparent(n; opacity = 0.8)
    return map(col -> RGBA(col, opacity), gen_colors(n))
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


function load_idf_sdf(nbehaviors, 
                      indfile = "expected_individual.jld2",
                      soc_root = "data/aggsocdf_")

    idf = load(indfile)["df"]
    sdf = load(soc_root * string(nbehaviors) * ".jld2")["aggsocdf"]

    return idf, sdf
end


# Calculates location where expected social and expected asocial payoffs 
# are equal, i.e., where social learning begins to do worse than asocial.
function calc_one_soc_ind_equal(idf, sdf, low_payoff, nbehaviors, steps_per_round; 
                                dthresh = 1e-4)


    sdflim = sdf[(sdf.low_payoff .== low_payoff) .&&
                 (sdf.steps_per_round .== steps_per_round),
                 :]

    # println(sdflim)
    
    ival = idf[(idf.low_payoff .== low_payoff) .&&
                  (idf.nbehaviors .== nbehaviors) .&&
                  (idf.steps_per_round .== steps_per_round), :].mean_prev_net_payoff
    ival /= steps_per_round
    # println(ival)

    svals = sdflim.mean_prev_net_payoff ./ steps_per_round
    # println(svals)

    d = svals .- ival
    # println(d)

    # If the data cross zero find the x-intercept using slope-intercept
    # formula for the line segment between datapoints where data first crosses
    # zero...
    diff_lt0 = d .< 0
    if count(diff_lt0) > 0
        
        firstnegindex = argmax(diff_lt0)
        @assert firstnegindex > 1 "Unexpected negative difference in first expected social value"

        lastposindex = firstnegindex - 1
        
        du = 0.1  # Resolution over environmental variability in the model.
        b = d[lastposindex]  # Impose axes over the two crossing datapoints.
        m = d[firstnegindex] - b  # Calculate slope over one unit of x.
        xint = -b / m
        
        u = sdflim.env_uncertainty
        # The point where u crosses zero is then the last u for which it was
        # non-negative, plus the x-intercept in terms of the variable 
        # substitution above, scaled appropriately to transform back to u.
        u_eq = u[lastposindex] + (xint * du)
        
    # ...otherwise find the first absolute difference within a threshold, 
    # incrementing the threshold until there is at least one difference
    # within threshold.
    else
        # Find where difference is less than a threshold, increasing threshold
        # by dthresh if no differences are within the threshold.
        thresh = dthresh
        under_thresh = abs.(d) .< thresh
        while sum(under_thresh) == 0
            thresh += dthresh
            # println("Trying threshold=$thresh")
            under_thresh = abs.(d) .< thresh 
        end

        u_eq = sdflim.env_uncertainty[argmax(under_thresh)]
        println("Final thresh: $thresh")
    end

    # return idf, sdf
    return u_eq
end


function load_steps_df(nbehaviors)

    return load("data/mainResult-yvar=step-B=$nbehaviors.jld2")["aggdf"]
end


# Returns a vector of u-values in order for each steps_per_round in stepsdf.
function calc_uGmax(stepsdf, low_payoff, nbehaviors)
    stepsdflim = stepsdf[stepsdf.low_payoff .== low_payoff, :]

    sdlim_argmax = combine(groupby(stepsdflim, :steps_per_round),  :step => argmax)

    uvec = sort(unique(stepsdflim.env_uncertainty))

    return map(smax -> uvec[smax], sdlim_argmax.step_argmax)
end


function plot_over_u_sigmoids(final_agg_df, nbehaviors, 
                                       yvar=:mean_social_learner; 
                                       low_payoffs=[0.1, 0.45, 0.8],
                                       figuredir=".", nfiles = 10, opacity = 0.8)
    df = final_agg_df

    if yvar == :mean_prev_net_payoff
        # nfiles = 100

        aggsoc_file = "data/aggsocdf_$nbehaviors.jld2"

        if isfile(aggsoc_file)
            println("Using synced expected social learner payoffs file $aggsoc_file")
            aggsocdf = load(aggsoc_file)["aggsocdf"]
        else
            socdf = load_random_df("data/sl_expected", nbehaviors, nfiles)

            aggsocdf = aggregate_final_timestep(socdf, yvar; socdf = true)

            @save aggsoc_file aggsocdf
        end

        # if nbehaviors ∈ [2, 4]
        #     filter!(r -> r.steps_per_round ∈ [1, 8], df)
        #     filter!(r -> r.steps_per_round ∈ [1, 8], aggsocdf)
        # else
        #     filter!(r -> r.steps_per_round ∈ [1, 20], df)
        #     filter!(r -> r.steps_per_round ∈ [1, 20], aggsocdf)
        # end
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
                if low_payoff == 0.8
                    ticloc = 200
                else
                    ticloc = 100
                end
                yticks = 0:ticloc:(ticloc * ceil(maximum(thisdf[!, yvar]) / ticloc))
            end

            # colorgenfn = yvar == :step ? gen_two_colors : gen_colors 

            # Calculate location where expected payoffs for all-social
            # and all-individual populations intersect.
            # soc_ind_expected_equal = calc_soc_ind_equal(low_payoff, nbehaviors)
            idf, sdf = load_idf_sdf(nbehaviors)
            sorted_steps = sort(unique(thisdf.steps_per_round))
            u_eq_locs = [
                calc_one_soc_ind_equal(idf, sdf, low_payoff, nbehaviors,
                                       steps_per_round)

                for steps_per_round in sorted_steps
            ]

            # println(u_eq_locs)

            stepsdf = load_steps_df(nbehaviors)
            u_Gmax_vec = calc_uGmax(stepsdf, low_payoff, nbehaviors)

            SEED_COLORS_TRANS = [RGBA(c, 0.7) for c in SEED_COLORS]

            if nbehaviors == 10
                colorkeypos = [.05w,0.275h]
            elseif nbehaviors == 4
                colorkeypos = [.05w,0.275h]
                # if low_payoff == 0.8
                #     colorkeypos = [.25w,0.275h]
                # end
            elseif nbehaviors == 2
                colorkeypos = [.865w,-0.205h]
            end

            if (nbehaviors == 2) && (low_payoff == 0.8)
                println("in here B=2")
                colorkeypos = [.735w,-0.205h]
                println(colorkeypos)
            end

            if (nbehaviors == 4) && (low_payoff == 0.8)
                println("in here B=4")
                # colorkeypos = [.685w,-0.225h]
                colorkeypos = [.25w,0.275h]
                println(colorkeypos)
            end
            println(colorkeypos)

            # Add some jitter to the x-intercepts.
            d = Normal(0.0, 0.01)
            u_eq_locs .+= rand(d, 4)
            u_Gmax_vec .+= rand(d, 4)

            u_eq_locs[u_eq_locs .> 1.0] .= 0.99
            u_Gmax_vec[u_Gmax_vec .> 1.0] .= 0.99

            p = plot(thisdf, x=:env_uncertainty, y=yvar, 
                     color = :steps_per_round, Geom.line, #Geom.point,

                     xintercept = [u_eq_locs..., u_Gmax_vec...],
                     Geom.vline(;color=[SEED_COLORS_TRANS..., 
                                        SEED_COLORS_TRANS...],
                                 style=[repeat([:ldash], 4)..., 
                                        repeat([:dot], 4)...],
                                 size=2.5pt),

                     Guide.xlabel(""),
                     Guide.ylabel(""), 
                     Guide.yticks(ticks=yticks),
                     # Scale.color_discrete(colorgenfn),
                     Scale.color_discrete(gen_colors),
                     Guide.colorkey(title="<i>L</i>", pos=colorkeypos),
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
            indiv_df = load(indiv_file)["df"]
            # XXX TODO need to get rid of this, but when I did code stopped 
            # working, but wasn't worth figuring out why at the time.
            if nbehaviors ∈ [2, 4]
                spr = [1, 2, 4, 8]
            else
                spr = [1, 5, 10, 20]
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

            lowpay_aggsocdf = 
                filter(r -> (r.low_payoff == low_payoff) &&
                            (r.steps_per_round ∈ spr),
                       aggsocdf)

            for r in eachrow(lowpay_aggsocdf)
                r.mean_prev_net_payoff /= convert(Float64, r.steps_per_round)
            end

            SEED_COLORS_TRANS = [RGBA(c, 0.8) for c in SEED_COLORS]

            p = plot(
                     layer(thisdf, x=:env_uncertainty, y=yvar,  
                           color = :steps_per_round, Geom.line, 
                           style(line_width=3.0pt)), # Geom.point),
                     yintercept = expected_individual_intercepts,
                     Geom.hline(; 
                                # color=[RGBA(sc, opacity) for sc in SEED_COLORS],
                                color=SEED_COLORS_TRANS,
                                # color=[SEED_COLORS[1],
                                #        SEED_COLORS[2],
                                #        SEED_COLORS[3],
                                #        SEED_COLORS[4]], 
                                style=:ldash, size=2.5pt),
                     layer(lowpay_aggsocdf, x=:env_uncertainty, y=yvar, Geom.line,
                           # Geom.point,
                           color = :steps_per_round, 
                           style(#$point_shapes=[diamond],
                           #       point_size=4.5pt, 
                                 line_width=2.5pt, 
                                 line_style=[:dot])), 
                     Guide.xlabel(""),
                     Guide.ylabel(""), 
                     # Guide.yticks(ticks=yticks),
                     # Scale.color_discrete(gen_two_colors),
                     Scale.color_discrete(_ -> SEED_COLORS_TRANS), #n -> gen_colors(n)),#; opacity = opacity)),
                     Guide.colorkey(title="<i>L</i>", pos=[.865w,-0.225h]),
                    # )
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


function payoffs_legend(; outfile = "testfig/payoffs_legend.pdf")

    xsims1 = 0.0
    xsims2 = 1.0
    xindiv1 = 4.25
    xindiv2 = 5.25
    xsoc1 = 8.75
    xsoc2 = 10.0

    y = 0

    set_default_plot_size((7*2.54)cm, 1.50cm)

    p = plot(
         Theme(minor_label_font_size = 0pt, # default_color=colorant"black", 
               grid_color = "white", ),

             layer(x=[xsims1, mean([xsims1, xsims2]), xsims2], 
               y=[y, y, y], Geom.line, color = [colorant"black"], #Geom.point, 
               style(#point_shapes=[Shape.circle],
                     #point_size=4.5pt, 
                     line_width=2.5pt)),
         Guide.annotation(compose(context(), 
                          text(xsims2 + 0.3, y, "Simulations", hleft, vcenter)
                         )),

         layer(x=[xindiv1, mean([xindiv1, xindiv2]), xindiv2], 
               y=[y, y, y], Geom.line, color = [colorant"black"],
               style(line_width=2.5pt, 
                     line_style=[:ldash])),
         Guide.annotation(compose(context(), 
                                  text(xindiv2 + 0.3, y, "All asocial", hleft, vcenter)
                         )),

         layer(x=[xsoc1, mean([xsoc1, xsoc2]), xsoc2], 
               y=[y, y, y], Geom.line, #Geom.point, 
               color = [colorant"black"],
               style(point_shapes=[diamond],
                     point_size=4.5pt, 
                     line_width=2.5pt, 
                     line_style=[:dashdot])),
         # text

         Guide.xticks(label=false, ticks=nothing),
         # Guide.yticks(label=false, ticks=[0]),
         Guide.xlabel(nothing), Guide.ylabel(nothing),
         Guide.annotation(compose(context(), 
                                  text(xsoc2 + 0.3, y, "All social", hleft, vcenter)
                         )),
         layer(x=[xsoc2+2.5], 
               y=[y], Geom.line, Geom.point,
               style(point_shapes=[Shape.circle],
                     point_size=0pt, 
                     line_width=2.5pt)),
        )
        
    draw(PDF(outfile), p)
end


function mean_social_legend(outfile = "papers/UncMod/Figures/mean_social_legend_lines.pdf")
    xmaxdrift1 = 0.0
    xmaxdrift2 = 1.0
    xsocasoceq1 = 2.15
    xsocasoceq2 = 3.15

    y = 0

    set_default_plot_size((7*2.54cm), 1.5cm)

    p = plot(
         Theme(minor_label_font_size = 0pt, # default_color=colorant"black", 
               grid_color = "white", ),

             layer(x=[xmaxdrift1, mean([xmaxdrift1, xmaxdrift2]), xmaxdrift2], 
               y=[y, y, y], Geom.line, color = [colorant"grey"], #Geom.point, 
               style(line_style=[:ldash], line_width=2.5pt)),
         Guide.annotation(compose(context(), 
                          text(xmaxdrift2 + 0.3, y, "", hleft, vcenter)
                         )),

         Theme(minor_label_font_size = 0pt, # default_color=colorant"black", 
               grid_color = "white", ),

             layer(x=[xsocasoceq1, mean([xsocasoceq1, xsocasoceq2]), xsocasoceq2], 
               y=[y, y, y], Geom.line, color = [colorant"grey"], #Geom.point, 
               style(line_style=[:dot],
                     line_width=2.5pt)),
         Guide.annotation(compose(context(), 
                          text(xsocasoceq2 + 0.3, y, "Social = Asocial", hleft, vcenter)
                         )),
         Guide.xticks(label=false, ticks=nothing),
         # Guide.yticks(label=false, ticks=[0]),
         Guide.xlabel(nothing), Guide.ylabel(nothing),
        )

    # XXX actually saves something without labels I'll import in Keynote and
    # annotate using Keynote's LaTeX functionality.

    draw(PDF(outfile), p)
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
    println("Loading data from files matching $globstring for aggregation")

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
