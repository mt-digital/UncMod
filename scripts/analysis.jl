using DataFrames
using Gadfly
using JLD2
# Not used here, but convenient for making notebooks.
import Cairo, Fontconfig

include("../src/experiment.jl")


PROJECT_THEME = Theme(
    point_size=6.5pt, major_label_font_size = 16pt, 
    minor_label_font_size = 14pt, key_title_font_size=14pt, 
    line_width = 2pt, key_label_font_size=14pt
)


function plot_series(
    result; 
    start_step = 0, 
    legendkeys = [:nbehaviors, :low_reliability, :high_reliability],
    outcomevar = :soclearnfreq,)
    
    if result isa String
        results_dict = load(result)

        if haskey(results_dict, "resdf_trialsmean")
            results_dict["result"] = results_dict["resdf_trialsmean"]
        end

        result = results_dict["result"]

        if haskey(results_dict, "models")
            models = results_dict["models"]
        end
    end

    # Create column that will be used for labeling each series.
    result.legendtuple = make_legend_tuple(result, legendkeys)

    legendtitle = join([string(key) for key in legendkeys], ", ")

    if @isdefined models
        title = make_title(models)
    else
        title =  legendtitle
    end

    if outcomevar == :soclearnfreq
        ylabel = "Social learning frequency"
    else
        ylabel = "% Optimal"
    end

    plot(
        filter(r -> r.step >= start_step, result),
        x = :step, y = outcomevar, color = :legendtuple,
        Geom.line, Theme(line_width=1.5pt),
        Guide.xlabel("Time step"), Guide.ylabel(ylabel),
        # Guide.colorkey(title=legendtitle)
        Guide.colorkey(title=""),
        # Guide.xticks(ticks=[1.0, 5.0, 9.0]), 
        Guide.yticks(ticks=0.1:0.1:1.0), 
        # Coord.cartesian(ymin=-0.05),
        Guide.title("Legend key: " * title)
     )
end

function payoffs_heatmap(
    result;
    agg_start_step = 80_000,
    zvar = :soclearnfreq,
    colormin = 0.45, colormax = 0.65)

    endtimesdf = filter(r -> r.step >= agg_start_step, result)

    endtimesgrouped = sort(combine(
        groupby(endtimesdf, [:low_reliability, :high_reliability]), 

        [:soclearnfreq, :pct_optimal] =>
        ((soclearnfreq, pct_optimal) ->
            (soclearnfreq = mean(soclearnfreq),
             pct_optimal = mean(pct_optimal))
        ) =>
        AsTable),

        [:low_reliability, :high_reliability]
    )
    
    # plotdata = endtimesgrouped
    # lowrels = vcat(unique(endtimesgrouped.low_reliability), 0.9)
    # highrels = vcat(0.1, unique(endtimesgrouped.high_reliability))

    lowrels = vcat(unique(endtimesgrouped.low_reliability), 0.9)
    highrels = vcat(1.0, unique(endtimesgrouped.high_reliability))
    nrels = length(lowrels)

    lowrels = vcat([repeat([lowrels[ii]], nrels) for ii in 1:nrels]...)
    highrels = repeat(highrels, nrels)

    plotdata = DataFrame(
        :π_high => highrels,
        :π_low => lowrels,
        :z => NaN
    )
    
    # println(endtimesgrouped)
    for row in eachrow(endtimesgrouped)
        rowselector = (plotdata.π_low  .== row.low_reliability) .& 
                      (plotdata.π_high .== row.high_reliability)
        plotdata[rowselector, :z] .= row[zvar]
    end

    z = reshape(plotdata[!, :z], nrels, nrels)

    xticklabels = Dict(zip(1:9, map(t -> string(t), 0.1:0.1:0.9)))
    yticklabels = Dict(zip(1:9, map(t -> string(t), reverse(0.1:0.1:0.9))))
    
    z = reverse(z, dims=1)

    if zvar == :soclearnfreq
        zlabel = "Social learning frequency"
    else
        zlabel = "% Optimal"
    end
    
    spy(z, 
        Scale.x_discrete(labels = x -> xticklabels[x]),
        Guide.xticks(orientation=:vertical),
        Guide.xlabel("π_low"), Guide.ylabel("π_high"),

        Scale.y_discrete(labels = y -> yticklabels[y]),

        Scale.color_continuous(minvalue=colormin, maxvalue=colormax),
        Guide.colorkey(title = zlabel),
        PROJECT_THEME)

end


function plot_final(result;
    agg_start_step = 80_000,
    xvar = :nbehaviors, yvar = :soclearnfreq, 
    legendkeys = [:low_reliability, :high_reliability],
    xticks = [5, 20, 100],
    yticks = 0.0:0.2:1.0,
    xlabel = nothing,
    title = nothing,
    legendtitle = nothing
    )

    if result isa String
        results_dict = load(result)

        if haskey(results_dict, "resdf_trialsmean")
            results_dict["result"] = results_dict["resdf_trialsmean"]
        end

        result = results_dict["result"]

        if haskey(results_dict, "models")
            models = results_dict["models"]
        end
    end

    # Make dataframe that takes mean yvar from agg_start_step to the end
    # using split/apply/combine. See line 30 in scripts/slurm/run_trials.jl.
    endtimesdf = filter(r -> r.step >= agg_start_step, result)

    endtimesgroupby = combine(
        groupby(endtimesdf, vcat(xvar, legendkeys)), 

        [:soclearnfreq, :pct_optimal] =>
        ((soclearnfreq, pct_optimal) ->
            (soclearnfreq = mean(soclearnfreq),
             pct_optimal = mean(pct_optimal))
        ) =>
        AsTable
    )
    
    # Create column that will be used for labeling each series.
    endtimesgroupby.legendtuple = make_legend_tuple(endtimesgroupby, legendkeys)

    if isnothing(legendtitle)
        legendtitle = join([string(key) for key in legendkeys], ", ")
    end

    if isnothing(xlabel)
        xlabel = string(xvar)
    end

    if isnothing(title)
        if @isdefined models
            title = make_title(models)
        else
            title =  legendtitle
        end
    end

    if yvar == :soclearnfreq
        ylabel = "Social learning frequency"
    else
        ylabel = "% Optimal"
    end

    plot(
        endtimesgroupby,
        x = xvar, y = yvar, color = :legendtuple,
        Geom.line, 
        Geom.point, 
        Theme(line_width=1.5pt),
        Guide.xlabel(xlabel), Guide.ylabel(ylabel),
        Guide.colorkey(title=legendtitle),
        Guide.xticks(ticks=xticks), 
        Guide.yticks(ticks=yticks),
        # Guide.yticks(ticks=0.45:0.05:0.7), 
        # Coord.cartesian(ymin=-0.05),
        Guide.title(title),
        shape=[Shape.diamond], 
        PROJECT_THEME
     )
        
end

function make_title(models)
    if models[1].selection_strategy == Softmax
        title = "τ = $(models[1].τ_init)\n$legendtitle"
    else
        title = "ϵ = $(models[1].ϵ_init)\n$legendtitle"
    end
end

function make_legend_tuple(result, legendkeys)

    map(
        # Need to replace some cruft introduced when making string of list
        # comprehension inside second replace.
        r -> replace(
            replace(
                string([r[el] for el in legendkeys]), 
                "[" => "("
            ), 
            "]" => ")"
        ),

        eachrow(result)
    )

end
