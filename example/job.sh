#!/bin/bash 
## Job Name 
#SBATCH --job-name=sn2_pm6
## Allocation Definition
#SBATCH --account=pfaendtner
#SBATCH --partition=pfaendtner
## Resources 
## Nodes 
#SBATCH --nodes=1
# SBATCH --ntasks-per-node=8
## Walltime (ten minutes) 
#SBATCH --time=3:00:00
# E-mail Notification, see man sbatch for options
#SBATCH --mail-type=NONE
## Memory per node 
#SBATCH --mem=20G
## Specify the working directory for this job 
#SBATCH --chdir=/gscratch/pfaendtner/lemmoi/transition-sampling/tmp/

# Import any modules here

# Scripts to be executed here
source /gscratch/pfaendtner/lemmoi/transition-sampling/sourceme.sh

python main.py

exit 0
