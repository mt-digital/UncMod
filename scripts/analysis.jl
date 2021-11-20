using DataFrames
using Gadfly
using JLD2


include("../src/experiment.jl")


function plot_series(
    result; 
    start_step = 0, 
    legendkeys = [:high_reliability, :low_reliability, :nbehaviors],
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

    result.legendtuple = map(
        r -> replace(
            replace(string([r[el] for el in legendkeys]), "Real[" => ""), 
            "]" => ""
        ), 
        eachrow(result)
    )

    legendtitle = join([string(key) for key in legendkeys], ", ")
    if @isdefined models
        if models[1].selection_strategy == Softmax
            title = "τ = $(models[1].τ_init)\n$legendtitle"
        else
            title = "ϵ = $(models[1].ϵ_init)\n$legendtitle"
        end
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





