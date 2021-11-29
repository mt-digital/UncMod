using DataFrames
using Gadfly
using JLD2


include("../src/experiment.jl")


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

function plot_final(result;
    agg_start_step = 80_000,
    xvar = :nbehaviors, yvar = :soclearnfreq, 
    legendkeys = [:low_reliability, :high_reliability],
    xticks = [5, 20, 100],
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
        Guide.yticks(ticks=0.45:0.05:0.7), 
        # Coord.cartesian(ymin=-0.05),
        Guide.title(title),
        shape=[Shape.diamond], Theme(point_size=4.5pt)
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
