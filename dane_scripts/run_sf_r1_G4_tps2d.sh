#!/bin/tcsh
##### These lines are for Slurm
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH -p pdebug
#SBATCH --mail-type=ALL
#SBATCH -o output.%j              #Output file name
###SBATCH -A bedonian1

### Job commands start here 
### Display some diagnostic information
echo '=====================JOB DIAGNOSTICS========================'
date
echo -n 'This machine is ';hostname
echo -n 'My jobid is '; echo $SLURM_JOBID
echo 'My path is:' 
echo $PATH
echo 'My job info:'
squeue -j $SLURM_JOBID
echo 'Machine info'
sinfo -s

echo '=====================JOB STARTING=========================='

srun  python3.11 sf_tps2d_uq_post.py /usr/workspace/bedonian1/tps2d_mf_post_r1_G4 /usr/workspace/bedonian1/tps2d_mf_r1_G4/  