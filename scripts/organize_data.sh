##
# Script to organize simulation output data and create figure output directories
# for analysis. The scripts in scripts/analysis.jl rely on this directory structure.
#
# Disclaimer: this is not the best way to handle this, and directory names evolved
# rather arbitrarily. Thank you for your patience and forgiveness!
# 
# Author: Matthew A. Turner <maturner01@gmail.com>
# Date: 2023-02-27


# Organize simulation data as expected by scripts/analysis.jl. Assumes main.zip,
# supplement.zip, and sl_expected.zip files have been downloaded and extracted
# in the `data/` directory under the project/repository directory.

# First create sensitivity data directories.
mkdir data/N_sensitivity/numagents={50,200}
mkdir data/nteachers_sensitivity/nteachers={2,20}
# In the paper we used softmax greediness (inverse temperature), but 
# in the code we used tau, the softmax temperature, and this is reflected in our
# data naming scheme.
mkdir data/tau/{0.01,1.0}

# Now move sensitivity analysis files to the appropriate directories.
mv data/N_sensitivity/*numagents=50* data/N_sensitivity/numagents=50
mv data/N_sensitivity/*numagents=200* data/N_sensitivity/numagents=200
mv data/nteachers_sensitivity/*nteachers=20* data/nteachers_sensitivity/nteachers=20
mv data/nteachers_sensitivity/*nteachers=2* data/nteachers_sensitivity/nteachers=2
mv data/tau/*tau=[0.01* data/tau/0.01
mv data/tau/*tau=[1.0* data/tau/1.0


# Create figure output directories.
mkdir -p main_figures \
    supplement_figures/{sensitivity_tau=0.01,sensitivity_tau=1.0,numagents=200,numagents=50,tau=0.01,tau=1.0}

