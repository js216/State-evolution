#!/bin/bash
#SBATCH --partition day
#SBATCH --job-name example_1D
#SBATCH --nodes 5
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --mem-per-cpu 20M
#SBATCH --time 00:30:00
#SBATCH --mail-type all
#SBATCH --mail-user jakob.kastelic@yale.edu
#SBATCH --output /home/fas/demille/jk2534/project/State-evolution/example/slurm-%j.out
#SBATCH --error /home/fas/demille/jk2534/project/State-evolution/example/slurm-%j.out
module load Python/3.6.4-foss-2018a
mpirun -n 4 --mca mpi_warn_on_fork 0 python3 /home/fas/demille/jk2534/project/State-evolution/src/main.py --info /home/fas/demille/jk2534/project/State-evolution/example example_1D.json
