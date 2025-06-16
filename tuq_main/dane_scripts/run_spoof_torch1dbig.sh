#!/bin/bash
#SBATCH -t 00:02:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p pdebug
##SBATCH --mail-type=ALL
#SBATCH -o output.%j              #Output file name
#SBATCH --signal=B:10@30         # Send signal 10 (SIG_USR1) 30s before time limit
###SBATCH -A bedonian1


if [ ! -f "%FINISH_REPLACE" ] ; then
	sbatch --dependency=afterany:$SLURM_JOBID run_spoof_torch1dbig.sh
else
	exit 0
fi

# creates automatic checkpoints and automatically starts from the most recent checkpoint
srun python3.11 spoof_torch1dbig.py %INPUT_REPLACE

touch %FINISH_REPLACE
exit 0