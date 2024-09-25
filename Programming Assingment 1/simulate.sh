#!/bin/bash
#SBATCH -p batch # batch, gpu, preempt, mpi or your group's own partition
#SBATCH --job-name=simulate #job name
#SBATCH --time=01-00:00:00 #requested time (DD-HH:MM:SS)
#SBATCH --nodes 1 #1 nodes
#SBATCH --ntasks 1 #2 tasks total (default 1 CPU core per task) = # of cores
#SBATCH --cpus-per-task 2
#SBATCH --mem=8G #requesting 2GB of RAM total
#SBATCH --output=%x-%J-%u.out #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=%x-%J-%u.err #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL
#SBATCH --mail-user=casey.owen@tufts.edu


# module load anaconda/2021.05
# First run: 
# conda activate /cluster/home/cowen03/cluster/home/cowen03/condaenv/capstoneenv
python main.py