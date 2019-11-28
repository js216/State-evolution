#!/bin/bash
#SBATCH --output dsq-2019-11-27_run2-%A_%1a-%N.out
#SBATCH --array 0-6
#SBATCH --job-name dsq-2019-11-27_run2
#SBATCH --nodes=1 --ntasks=1 --mem-per-cpu=500M --time=12:00:00 --mail-type=ALL --mail-user=jakob.kastelic@yale.edu --cpus-per-task=32

# DO NOT EDIT LINE BELOW
/gpfs/loomis/apps/avx/software/dSQ/0.96/dSQBatch.py /gpfs/loomis/project/demille/jk2534/State-evolution/slurm/2019-11-27_run2.txt /gpfs/loomis/project/demille/jk2534/State-evolution/slurm

