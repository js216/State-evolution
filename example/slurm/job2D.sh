#!/bin/bash
#SBATCH --partition       day
#SBATCH --job-name        example_2D
#SBATCH --nodes           1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task   32
#SBATCH --mem-per-cpu     1G
#SBATCH --time            12:00:00
#SBATCH --mail-type       ALL
#SBATCH --mail-user       jakob.kastelic@yale.edu

prog=/home/fas/demille/jk2534/project/State-evolution/src/main.py
run_dir=/home/fas/demille/jk2534/project/State-evolution/example
options_file=example_2D.json

module load miniconda
source activate tf_gpu

#mpirun -n 1 python3 -m cProfile -s tottime $prog $run_dir $options_file
#mpirun -n 1 python3 $prog $run_dir $options_file
python3 $prog $run_dir $options_file
