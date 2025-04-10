import os
import yaml
from subprocess import run
from mpi4py import MPI
import sys

from sample_utils import *


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


"""
Script to run torch simulations for many samples

Will at some point be able to run sample groups across different models for MLBLUE

torch1d:

python torch1d.py input_file.yml

"""

# r7
home = os.environ["HOME"]
# sample_dir = f"{home}/bedonian1/torch1d_samples_r3"
# sample_dir = f"{home}/bedonian1/torch1d_samples_r3_no_4p_to_h"
# sample_dir = f"{home}/bedonian1/torch1d_samples_r3_no_4p_to_h_dt"
# sample_dir = f"{home}/bedonian1/torch1d_samples_r6_3" # initial state evolved from the mean of all rate samples after 0.015 s, to 0.115 s
sample_dir = f"{home}/bedonian1/torch1d_samples_r7_1" # initial state evolved from the mean of all rate samples after 0.015 s, to 0.115 s
# template_file = f"{home}/torch-sensitivity/trevilo-cases/torch_7sp_chem/nominal/axial_icp_torch.yml" # keep this to deal with restarts
# template_file = f"{home}/bedonian1/nominal_r6/torch1d_input_r.yml" # keep this to deal with restarts
template_file = f"{home}/bedonian1/mean_r6/torch1d_input_r.yml" # keep this to deal with restarts

if len(sys.argv) > 1:
    sample_dir = sys.argv[1]

# fstep_over = None
fstep_over = 70000 #override, this should be the final time step

torch1d_exec = f"{home}/torch1d/torch1d.py"
restart = True # enable restart from stopped state
pcomm = 'python3.11'

######### Retrieve baseline information from template in case of restarts
with open(template_file) as f:
    torch1d_in_temp = yaml.safe_load(f)
    
# get total number of time steps
fstep = torch1d_in_temp['time_integration']['number_of_timesteps']
ic = torch1d_in_temp['state']['initial_condition']
prefix = torch1d_in_temp['prefix']
if fstep_over is not None:
    fstep = fstep_over

######### Loop through sample directories in sample_dir
samples = os.listdir(sample_dir)
samples.sort()

N = len(samples)

cases = divide_cases(N, size)

for isamp in cases[rank]:
# for r6, use 1735 to test
# for isamp in [1]:

    sample = samples[isamp]

    # TODO: In each sample, detect which models to run
    # For now, just run torch1d based on yml file presence

    dir = sample_dir + '/' + sample 

    run_torch1d = False
    # run_tps, etc.

    # check if there exists a yaml file (and read its contents)
    for ifname in os.listdir(dir):
        if ifname.endswith('.yml'):
            run_torch1d = True
            torch1d_infile = ifname
            break
            
    

    # check if calculation already performed
    if os.path.isdir(dir + '/output/'):
        for fname in os.listdir(dir + '/output/'):
            if fname.endswith(f'-{fstep:08d}.h5'): 
                run_torch1d = False
                break

        # now check where the calculation stopped
        if run_torch1d and len(os.listdir(dir + '/output/')):
            with open(dir + '/' + torch1d_infile) as f:
                torch1d_in = yaml.safe_load(f)

            if restart:
                cstep = 0
                fname_s = prefix + '-00000000.h5'
                for fname in os.listdir(dir + '/output/'):
                    if not fname.endswith('crashed.h5'):
                        cand = int(fname[-11:-3])
                        if cand > cstep:
                            cstep = cand
                            fname_s = fname

                # need to adjust the input file
                fstep_s = fstep - cstep
                ic_s = dir + '/output/' + fname_s

                torch1d_in['time_integration']['number_of_timesteps'] = fstep_s
                torch1d_in['state']['initial_condition'] = ic_s

                torch1d_infile_r = 'r_' + torch1d_infile
                with open(dir + '/' + torch1d_infile_r, 'w') as f:
                    yaml.safe_dump(torch1d_in, f)

        else:
            torch1d_infile_r = torch1d_infile
    else:
        torch1d_infile_r = torch1d_infile
    if run_torch1d:
        # for fname in os.listdir(dir):
        #     if fname.endswith('.yml'):
        #         torch1d_in = fname

        if restart:
            inputfile = dir + '/' + torch1d_infile_r
        else:
            inputfile = dir + '/' + torch1d_infile
        

        # run torch1d
        run([pcomm, "torch1d.py", inputfile])

