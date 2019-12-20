#!/bin/bash
#SBATCH --partition       day
#SBATCH --job-name        example_1D
#SBATCH --nodes           5
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task   10
#SBATCH --mem-per-cpu     20M
#SBATCH --time            00:30:00
#SBATCH --mail-type       ALL
#SBATCH --mail-user       jakob.kastelic@yale.edu

prog=/home/fas/demille/jk2534/project/State-evolution/src/main.py
run_dir=/home/fas/demille/jk2534/project/State-evolution/example
options_file=pickle_test.json
mpi_params="--mca mpi_warn_on_fork 0"

#module load miniconda
#source activate tf_gpu
module load Python/3.6.4-foss-2018a

#mpirun -n 1 python3 -m cProfile -s tottime $prog $run_dir $options_file
#python3 $prog $run_dir $options_file
mpirun -n 3 $mpi_params python3 $prog $run_dir $options_file
