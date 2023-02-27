# UncMod: Model supporting "The Form of Uncertainty Affects Selection for Social Learning"

It is a common truism that social learning evolved to help overcome uncertainty. This is
often summarized as _copy when uncertain_. Yet, it was not clear
what forms of uncertainty are most important or how different forms of
uncertainty interact to affect the evolution of social learning. 

We developed this model to address the need for an improved systematic
understanding of the relationship between varieties of uncertainty and the
evolution of social learning. We introduce and analyze this model in our article
currently under review, with [preprint available on SocArXiV](https://osf.io/preprints/socarxiv/brqmn/).


## Installation and setup.

To (locally) reproduce this project, do the following:

1. [Download](https://github.com/mt-digital/UncMod/archive/refs/heads/main.zip) 
or clone the repository (`git clone https://github.com/mt-digital/UncMod`)
2. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.


## The model

Our analysis is based on running ensembles of models initialized with systematically varied uncertainty and other parameters.
To run one single model and observe the agent-level output dataframe, run the following in the terminal, which uses the [`run!` function from Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/tutorial/#Agents.run!) (note default kwarg model parameter settings given in `uncertainty_learning_model` definition):

```julia
# Activate our working environment that was initialized in Installation steps above.
using Pkg; Pkg.activate(".")

# Load model code.
include("src/model.jl")

# Initialize a new model, not yet run, more setup to do.
m = uncertainty_learning_model()

# Set which columns we want and how to aggregate them for agent-level data...
adata = [(:behavior, countmap), (:social_learner, mean), (:prev_net_payoff, mean)]

# ...and for model parameters/data.
mdata = [:env_uncertainty, :trial_idx, :high_payoff,
         :low_payoff, :nbehaviors, :steps_per_round, :optimal_behavior]

# Now run the model for 100 time steps.
nsteps = 100
agents_df, model_df = run!(m, agent_step!, model_step!, nsteps; adata, mdata)

println(agents_df)
```

This should print something like this:

<img width="500" alt="output screenshot" src="https://user-images.githubusercontent.com/2425472/197694394-d9d0d6bc-347e-42bc-b636-0cc7b9ab2d84.png">


## Small-scale computational experiments run locally

Agents.jl provides the [`ensemblerun!` function](https://juliadynamics.github.io/Agents.jl/stable/tutorial/#Agents.ensemblerun!) for running ensembles of models. We wrap the `ensemblerun!` function to run our computational experiments using the `experiment` function in [`src/experiment.jl`](src/experiment.jl). This works by first initializing a list of models to pass to `ensemblerun!` initialized with the Cartesian product of individual uncertainty parameter settings.

```julia
agentdf, modeldf = experiment(2; env_uncertainty=[0.0,0.5,1.0], steps_per_round=[1,2])
```

## Computational experiments on a Slurm cluster

For this project, we used the [Sherlock cluster at Stanford University](https://www.sherlock.stanford.edu/) to
run our simulations at scale. The main data can be created by running [`./main.sh`](main.sh) from the root directory on
a Slurm cluster. Examine `main.sh` to see which Slurm scripts are executed. Each Slurm script runs
[`scripts/run_analysis.jl`](scripts/run_analysis.jl), using the user-provided parameters for the given partition of the input parameter space.
`scripts/run_analysis.jl` reads command-line arguments, then sets up and runs computational experiments using code from [`src/experiment.jl`](src/experiment.jl).

To perform all computational experiments we analyzed, run also [`sl_expected.sh`](sl_expected.sh), which runs the homogenous all-social-learner 
simulations used in Figures 4 and 5 of the preprint. The sensitivity analyses are run with their corresponding shell scripts, [`nagents.sh`](nagents.sh), [`nteachers.sh`](nteachers.sh), and [`tau.sh`](tau.sh) (the model uses softmax temperature, $\tau$, not inverse temperature, i.e., "greediness", $\beta = 1 / \tau$).


## Analysis

Building the analysis figures is a multi-step progress that could have
benefitted from slightly more automation, but of course premature optimization
is always to be avoided, and the following was good enough for developing
publication-ready figures. Part of the complications came from trying to use
Julia to make all figures, when, in my opinion, `ggplot2` (or possibly any
approach using R) provides much more useful tools that could have made this
easier.

So, building all the figures requires a few steps, which are described below.

### Transfer or download datasets

If you ran the simulations on a cluster, make zip files of all outputs and
use `scp` to transfer those to your local machine, then skip ahead and follow
the instructions for organizing the output data, setting up output directories,
and running the analysis scripts.

To download the datasets, please get them from our associated [Open Science Foundation (OSF)
repository](https://osf.io/8kf7s). These should be saved to the `data` directory
which is included in this GitHub repository, and is empty except for an empty
file called ".include" that we use to include an empty directory in the repo. 
The three files to download from OSF are

- `main.zip`
- `sensitivity.zip`
- `sl_expected.zip`

When these are downloaded, use a terminal to navigate to the `data` directory,
then unzip the files:

```
unzip main.zip && unzip sensitivity.zip && unzip sl_expected.zip
```

### Organize supplemental data and create output figure directories

Use the script `scripts/organize_data.sh` to organize the supplemental data into
appropriately-named directories that are expected by the analysis scripts. Run
the following in a terminal command line

```
./scripts/organize_data.sh
```

Supplemental data directory names are not intuitive, and the names evolved rather arbitrarily
as the project went on. Sorry for this, but to see how this works, 
please view [the script](scripts/organize_data.sh) listed above.

### Run the analyses

Now the data have been prepared, and output directories created, run
another script that does all main and supplemental analyses,
[scripts/run_all_analyses.jl](scripts/run_all_analyses.jl):

```
julia scripts/run_all_analyses.jl
```

When this is first
run, it will take a long time to aggregate the main datasets since these
captured every time step in order to measure strength of selection (Figure 4 in
the paper). When the script first runs it creates an aggregated data file that
will be loaded on subsequent analysis runs. While this data aggregation runs 
you will see the following messages:

```
Loading data from files matching data/main/*nbehaviors=[2* for aggregation
Loading data from files matching data/main/*nbehaviors=[4* for aggregation
Loading data from files matching data/main/*nbehaviors=[10* for aggregation
```

Unfortunately there are further steps still to using these figures to create the figures found in our paper. Please see the LaTeX source and `makefile` build script in our paper repository: https://github.com/mt-digital/UncMod-JournalArticle

## Tests

Unit tests ensure the model is working properly and provide documentation of
the model API and functionality. To run the tests, `include` the test script from a
Julia REPL:

```
julia> using Pkg; Pkg.precompile(); include("src/test/runtests.jl")
```
