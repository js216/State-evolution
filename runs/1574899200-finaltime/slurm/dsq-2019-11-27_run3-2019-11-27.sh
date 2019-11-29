#!/bin/bash
#SBATCH --output dsq-2019-11-27_run3-%A_%1a-%N.out
#SBATCH --array 0-1
#SBATCH --job-name dsq-2019-11-27_run3
#SBATCH --nodes=1 --ntasks=1 --mem-per-cpu=500M --time=7-00:00:00 --mail-type=ALL --mail-user=jakob.kastelic@yale.edu --cpus-per-task=16 --partition=week

# DO NOT EDIT LINE BELOW
/gpfs/loomis/apps/avx/software/dSQ/0.96/dSQBatch.py /gpfs/loomis/project/demille/jk2534/State-evolution/slurm/2019-11-27_run3.txt /gpfs/loomis/project/demille/jk2534/State-evolution/slurm

