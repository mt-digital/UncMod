# UncMod: Model supporting "Different forms of uncertainty differently affect the evolution of social learning"

It is a common truism that social learning evolved to help individuals identify
beneficail behaviors in their environment when there is uncertainty. This is
often summarized as the _copy when uncertain_ strategy. Yet it is not clear
what forms of uncertainty are most important or how different forms of
uncertainty interact to affect the evolution of social learning. 

We developed this model to address this need for improved systematic
understanding of the relationship between varieties of uncertainty and the
evolution of social learning.


## Installation and setup.

To (locally) reproduce this project, do the following:

0. [Download](https://github.com/mt-digital/UncMod/archive/refs/heads/main.zip) 
or clone the repository (`git clone https://github.com/mt-digital/UncMod`)
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.


## Quickstart: run the model

Our analysis is based on running ensembles of models initialized with systematically varied uncertainty and other parameters.
a

## Advanced: run computational experiments on a Slurm cluster

For this project, we used the [Sherlock cluster at Stanford University](https://www.sherlock.stanford.edu/) to
run our simulations at scale. 

### scripts/run_trials.jl



## Tests

Unit tests ensure the model is working properly and provide documentation of
the model API and functionality. To run the tests, run the test script from a
terminal:

```
julia src/test/test_model.jl
```
