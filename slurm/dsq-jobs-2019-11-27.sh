#!/bin/bash
#SBATCH --output dsq-jobs-%A_%1a-%N.out
#SBATCH --array 0-7
#SBATCH --job-name dsq-jobs
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=24 --mem-per-cpu=250M --time=24:00:00 --mail-type=ALL --mail-user=jakob.kastelic@yale.edu

# DO NOT EDIT LINE BELOW
/gpfs/loomis/apps/avx/software/dSQ/0.96/dSQBatch.py /gpfs/loomis/project/demille/jk2534/State-evolution/slurm/jobs.txt /gpfs/loomis/project/demille/jk2534/State-evolution/slurm

