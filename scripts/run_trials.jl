using Distributed
using Dates

using DrWatson
quickactivate("..")

# Set up DrWatson to include vectors in autogen save names.
da = DrWatson.default_allowed(Any)
DrWatson.default_allowed(c) = (da..., Vector)

using ArgParse

using Comonicon
using JLD2

include("../src/experiment.jl")


s = ArgParseSettings()

function vecparse(T, s::AbstractString)

    if occursin(":", s)
        vals = split(s, ":")
        parseT(x) = parse(T, x)

        return parseT(vals[1]):parseT(vals[2]):parseT(vals[3])
        
    else
        s = replace(s, "[" => "")
        s = replace(s, "]" => "")

        return [parse(T, el) for el in split(s, ",")]
    end
end

# Define functions to parse vectors of floats...
function ArgParse.parse_item(::Type{Vector{Float64}}, s::AbstractString)
    vecparse(Float64, s) 
end

# ...and vectors of ints. Could not get templated version to work so had to dup.
function ArgParse.parse_item(::Type{Vector{Int64}}, s::AbstractString)
    vecparse(Int64, s)
end

function parse_cli()

    @add_arg_table s begin
    
        "experiment"
            help = "The type of experiment to run: 'expected-payoffs', 'steps-per-round', or 'nbehaviors'."
            arg_type = String
            required = true

        "datadirname"
            help = "Where to save experiment data within data directory"
            arg_type = String
            required = true

        "--niter"
            help = "Total number of iterations per simulation"
            arg_type = Int
            default = 100_000

        "--ntrials"
            help = "Number of trial simulations to run for this experiment"
            arg_type = Int
            default = 100

        "--vertical", "-v"
            help = "Flag to include vertical transmission in simulation."
            action = :store_true

        "--env_uncertainty", "-u"
            help = "Probability optimal behavior switches after a round/generation."
            arg_type = Vector{Float64}
            default = [0.0]

        "--nbehaviors"
            help = "Number of behaviors environment affords."
            arg_type = Vector{Int}

        "--steps_per_round"
            help = "Number of time steps per round/generation."
            arg_type = Vector{Int}
        
        "--low_payoff"
            help = "Low payoffs to include in experiment"
            arg_type = Vector{Float64}

        "--high_payoff"
            help = "Low payoffs to include in experiment"
            arg_type = Vector{Float64}
    end

    return parse_args(s)
end

function run_trials(ntrials = 100; 
                    outputfilename = "trials_output.jld2", 
                    experiment_kwargs...)

    println(experiment_kwargs)

    tic = now()

    adf, mdf, models = experiment(ntrials; experiment_kwargs...)

    adf.pct_optimal = map(
        r -> (haskey(r.countbehaviors_behavior, 1) ? 
                r.countbehaviors_behavior[1] : 
                0.0 )  / length(models[1].agents), 
        eachrow(adf)
    )

    resdf = innerjoin(adf,
                      mdf, 
                      on = [:ensemble, :step])

    result = combine(
        # Groupby experimental variables...
        groupby(resdf, [:step, :nbehaviors, :low_payoff, :high_payoff, 
                        :env_uncertainty, :payoff_variance, :steps_per_round]),

        # ...and aggregate by taking means over outcome variables, convert to table.
        [:mean_soclearnfreq, :mean_vertical_transmag, :pct_optimal] 
            =>
                (
                    (soclearnfreq, vertical_transmag, pct_optimal) -> 
                        (soclearnfreq = mean(soclearnfreq),
                         vertical_transmag = mean(vertical_transmag),
                         pct_optimal = mean(pct_optimal))
                ) 
            =>
                AsTable
    )

    @save outputfilename result

    trialstime = Dates.toms(now() - tic) / (60.0 * 1000.0)

    println("Ran expected payoffs trials in $trialstime minutes")

end

function main()
    parsed_args = parse_cli()
    println("Simulation run with following arguments:")
    for (arg, val) in parsed_args
        println("    $arg => $val")
    end
    
    # Depending on the experiment, ignore certain name keys.
    # experiment = parsed_args["experiment"]
    # rmkeys = ["experiment"]
    experiment = pop!(parsed_args, "experiment")
    datadirname = pop!(parsed_args, "datadirname") 

    # Make a copy of parsed args for use in naming output.
    nameargs = copy(parsed_args)

    rmkeys = []
    if experiment == "expected-payoff"
        rmkeys = [rmkeys..., "low_payoff", "high_payoff"]
    end

    for rmkey in rmkeys
        delete!(nameargs, rmkey)
    end

    outputfilename = savename(experiment, parsed_args, "jld2")
    outputfilename = replace(datadir(datadirname, outputfilename), " " => "")

    ntrials = pop!(parsed_args, "ntrials")

    pa_symbkeys = Dict(Symbol(key) => value for (key, value) in parsed_args)

    run_trials(ntrials; 
               outputfilename = outputfilename, 
               pa_symbkeys...)
end

main()


# run_trials(10; niter = 100_000, transledger = false, outputfilename = "softmax_novertical.jld2")
# run_trials(10; niter = 100_000, transledger = true, outputfilename = "vertical.jld2")
