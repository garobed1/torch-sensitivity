#!/bin/bash
#SBATCH -N 1
#SBATCH -n 112
#SBATCH -t 1:00:00
#SBATCH -p pdebug
#SBATCH --mail-type=ALL
#SBATCH -o output.%j              #Output file name
#SBATCH --signal=B:10@300        # Send signal 10 (SIG_USR1) 300s before time limit
##SBATCH -A bedonian1



srun ~/tps/src/tps --runFile tps_axi2d_input.ini

exit 0