##
# prep_sensitivities.sh: A script to prepare UncMod data for plotting and building
# into manuscript figure directories.
#
# Author: Matthew A. Turner <maturner01@gmail.com>
# Date: October 11, 2022
#


# Make subdirectories for numagents sensitivity.
cd data/N_sensitivity
mkdir "numagents=50" "numagents=200"

# Make subdirectories for nteachers sensitivity.

# Make subdirectories for tau sensitivity.
cd data/N_sensitivity
mkdir "tau=0.01" "tau=1.0"
