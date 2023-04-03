using Test

using DrWatson
quickactivate("../../")

include("../model.jl")

include("test_model.jl")
include("test_behavior_selection.jl")
