using Distributed
using Dates
using UUIDs: uuid4

using DrWatson
quickactivate("..")

# Set up DrWatson to include vectors in autogen save names.
da = DrWatson.default_allowed(Any)
DrWatson.default_allowed(c) = (da..., Vector)

using ArgParse

using Comonicon
using JLD2

# include("run_trials.jl")
include("expected_homogenous_payoff.jl")

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
    
        "datadirname"
            help = "Where to save experiment data within data directory"
            arg_type = String
            required = true

        "--env_uncertainty", "-u"
            help = "Probability optimal behavior switches after a round/generation."
            arg_type = Vector{Float64}
            default = collect(0.0:0.1:1.0)

        "--ntrials"
            help = "Number of trial simulations to run for this experiment"
            arg_type = Int
            default = 10

        "--nbehaviors"
            help = "Number of behaviors environment affords."
            arg_type = Vector{Int}
            default = [2,4]

        "--steps_per_round"
            help = "Number of time steps per round/generation."
            arg_type = Vector{Int}
            default = [1,2,4,8]
        
        "--low_payoff"
            help = "Low payoffs to include in experiment"
            arg_type = Vector{Float64}
            default = [0.1, 0.45, 0.8]

        "--high_payoff"
            help = "High payoffs to include in experiment"
            arg_type = Vector{Float64}
            default = [0.9]

        "--tau"
            help = "Softmax temperature"
            arg_type = Vector{Float64}
            default = [0.1]
    end

    return parse_args(s)
end


function run_trials(ntrials = 20; 
                    outputfilename = "test_social_payoff.jld2", 
                    experiment_kwargs...)

    tic = now()

    println("Starting trials at $(replace(string(tic), "T" => " "))")

    expected_social_joined_df = expected_social_payoff(; experiment_kwargs...)

    println("About to save!!!")

    @save outputfilename expected_social_joined_df

    trialstime = Dates.toms(now() - tic) / (60.0 * 1000.0)

    println("Ran expected payoffs trials in $trialstime minutes")
end


function main()
    parsed_args = parse_cli()

    # Create UUID for multiple runs with same parameters.
    println(parsed_args)
    parsed_args["id"] = string(uuid4())
    println("Simulation run with following arguments:")
    for (arg, val) in parsed_args
        println("    $arg => $val")
    end
    
    # Depending on the experiment, ignore certain name keys.
    datadirname = pop!(parsed_args, "datadirname") 

    # Make a copy of parsed args for use in naming output.
    nameargs = copy(parsed_args)

    outputfilename = savename(parsed_args, "jld2")
    outputfilename = replace(datadir(datadirname, outputfilename), " " => "")

    ntrials = pop!(parsed_args, "ntrials")
    pop!(parsed_args, "id")

    pa_symbkeys = Dict(Symbol(key) => value for (key, value) in parsed_args)

    run_trials(ntrials; 
               outputfilename = outputfilename, 
               pa_symbkeys...)
end

main()
