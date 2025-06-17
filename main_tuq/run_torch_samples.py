import os
import yaml
from subprocess import run
from file_util import t1dRestart
from mpi4py import MPI
import sys

from util_tuq.sample_utils import *


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()






"""
Script to run torch simulations for many samples

Will at some point be able to run sample groups across different models for MLBLUE

torch1d:

python torch1d.py input_file.yml

"""


home = os.environ["HOME"]



### Torch1D Sample Directory
# sample_dir = f"{home}/bedonian1/torch1d_samples_r3"
# sample_dir = f"{home}/bedonian1/torch1d_samples_r3_no_4p_to_h"
# sample_dir = f"{home}/bedonian1/torch1d_samples_r3_no_4p_to_h_dt"
# sample_dir = f"{home}/bedonian1/torch1d_samples_r6_3" # initial state evolved from the mean of all rate samples after 0.015 s, to 0.115 s
sample_dir = f"{home}/bedonian1/torch1d_samples_r7_1" # initial state evolved from the mean of all rate samples after 0.015 s, to 0.115 s

### Torch1D Input File Template
# template_file = f"{home}/torch-sensitivity/trevilo-cases/torch_7sp_chem/nominal/axial_icp_torch.yml" # keep this to deal with restarts
# template_file = f"{home}/bedonian1/nominal_r6/torch1d_input_r.yml" # keep this to deal with restarts
template_file = f"{home}/bedonian1/mean_r6/torch1d_input_r.yml" # keep this to deal with restarts



### Manually set the final time step 
# fstep_over = None
fstep_over = 70000 #override, this should be the final time step


### Command Line Override
if len(sys.argv) > 1:
    sample_dir = sys.argv[1]
if len(sys.argv) > 3:
    sample_dir = sys.argv[1]
    template_file = sys.argv[2]
    fstep_over = int(sys.argv[3])

### Torch1D Run Commands
torch1d_exec = f"{home}/torch1d/torch1d.py"
restart = True # enable restart from stopped state
pcomm = 'python3.11'





##########################################################################################################
# Script Starts Here
##########################################################################################################


def listdir_nopickle(path):
    return [f for f in os.listdir(path) if not f.endswith('.pickle')]



######### Retrieve baseline information from template in case of restarts
with open(template_file) as f:
    torch1d_in_temp = yaml.safe_load(f)
    
# get total number of time steps
fstep = torch1d_in_temp['time_integration']['number_of_timesteps']
ic = torch1d_in_temp['state']['initial_condition']
prefix = torch1d_in_temp['prefix']
if fstep_over is not None:
    fstep = fstep_over

# Loop through sample directories in sample_dir
# samples = os.listdir(sample_dir)
samples = listdir_nopickle(sample_dir)
samples.sort()


# NOTE: DEBUG
# samples = samples[-512:]
# samples = samples[-512:-256]
# samples = samples[11400:11776]
# remain = [1, 2, 5, 6, 9, 10, 14, 17, 21, 22, 25, 26, 29, 30, 33, 34, 37, 38, 41, 42, 46, 49, 53, 54, 57, 58, 61, 62, 65, 66, 69, 70, 73, 74, 78, 81, 85, 86, 89, 90, 93, 94, 97, 98, 101, 102, 105, 106, 110, 113, 117, 118, 121, 122, 126, 129, 133, 134, 137, 138, 142, 145, 149, 150, 153, 154, 157, 158, 161, 162, 165, 166, 169, 170, 174, 177, 181, 182, 185, 186, 190, 193, 194, 197, 198, 201, 202, 206, 209, 213, 214, 217, 218, 221, 222, 225, 226, 229, 230, 233, 234, 238, 241, 245, 246, 249, 250, 253, 254]
# # remain = [11653,11658,11669,11685,11701,11706,11717,11722,11733,11738,11749,11754,11765,11770]
# s2 = []
# for x in remain:
#     s2.append(samples[x])
# samples = s2


N = len(samples)
cases = divide_cases(N, size)

for isamp in cases[rank]:
# for r6, use 1735 to test
# for isamp in [1]:

    sample = samples[isamp]

    # TODO: In each sample, detect which models to run
    # For now, just run torch1d based on yml file presence

    sdir = sample_dir + '/' + sample 

    run_torch1d = False
    # run_tps, etc.

    # check if there exists a yaml file (and read its contents)
    for ifname in os.listdir(sdir):
        if ifname.endswith('.yml'):
            run_torch1d = True
            torch1d_infile = ifname
            break
            
    

    # check if calculation already performed
    if os.path.isdir(sdir + '/output/'):
        torch1d_infile_r, run_torch1d = t1dRestart(sdir, fstep, torch1d_infile)
    else:
        torch1d_infile_r = torch1d_infile

    if run_torch1d:
        # for fname in os.listdir(dir):
        #     if fname.endswith('.yml'):
        #         torch1d_in = fname

        if restart:
            inputfile = sdir + '/' + torch1d_infile_r
        else:
            inputfile = sdir + '/' + torch1d_infile
        

        # run torch1d
        run([pcomm, "torch1d.py", inputfile])


# comm.Barrier()
