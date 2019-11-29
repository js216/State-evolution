#!/bin/bash
#SBATCH --output dsq-2019-11-28_run1-%A_%1a-%N.out
#SBATCH --array 0-7
#SBATCH --job-name dsq-2019-11-28_run1
#SBATCH --cpus-per-task=35 --mem-per-cpu=300M --time=5:00:00 --mail-type=ALL --mail-user=jakob.kastelic@yale.edu

# DO NOT EDIT LINE BELOW
/gpfs/loomis/apps/avx/software/dSQ/0.96/dSQBatch.py /gpfs/loomis/project/demille/jk2534/State-evolution/slurm/2019-11-28_run1.txt /gpfs/loomis/project/demille/jk2534/State-evolution/slurm

