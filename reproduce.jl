using Distributed
using Dates
using UUIDs: uuid4

using DrWatson
quickactivate("..")

# Set up DrWatson to include vectors in autogen save names.
da = DrWatson.default_allowed(Any)
DrWatson.default_allowed(c) = (da..., Vector)

using ArgParse

# using Comonicon


include("../src/experiment.jl")


s = ArgParseSettings()


function parse_cli()
    @add_arg_table s begin
        "command"
            help = 
                "Specify command to execute for reproducing part or all analyses"
            arg_type = String
            required = true

    end
end
