import os
import yaml
from subprocess import run
from file_util import t1dRestart
import sys

from sample_utils import *


from mpi4py import MPI

"""
Script to compress individual torch1d time solution files into single tar balls
to avoid hitting system inode limits 


"""


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()




### Input Directories
home = os.environ["HOME"]
# torch_dir = f"{home}/bedonian1/torch1d_samples_r7" # initial state evolved from the mean of all rate samples after 0.015 s, to 0.115 s
# torch_dir = f"{home}/bedonian1/torch1d_samples_r7_1" # initial state evolved from the mean of all rate samples after 0.015 s, to 0.115 s
# torch_dir = f"{home}/bedonian1/torch1d_resample_r7_coarse" # initial state evolved from the mean of all rate samples after 0.015 s, to 0.115 s
# torch_dir = f"{home}/bedonian1/torch1d_resample_r7" # initial state evolved from the mean of all rate samples after 0.015 s, to 0.115 s
torch_dir = f"{home}/bedonian1/torch1d_r1_G2_mid" 
# torch_dir = f"{home}/bedonian1/torch1d_resample_sens_r8/" 
template_file = f"{home}/bedonian1/mean_r6/torch1d_input_r.yml" # keep this to deal with restarts

if len(sys.argv) > 1:
    sample_dir = sys.argv[1]

### Final Time Step, Set Manually
fstep_over = 65000 #override, this should be the final time step
restart = True # enable restart from stopped state
pcomm = 'python3.11'






##########################################################################################################
# Script Starts Here
##########################################################################################################


def listdir_nopickle(path):
    return [f for f in os.listdir(path) if not f.endswith('.pickle') and not f.endswith('.tar.gz')]


samples = listdir_nopickle(torch_dir)
samples.sort()

N = len(samples)
cases = divide_cases(N, size)

# N = len(samples)
for isamp in cases[rank]:
# for isamp in [17852, 18049]:

    sample = samples[isamp]
    sdir = torch_dir + '/' + sample + '/output/'

    slist = listdir_nopickle(sdir)

    try:
        j = next(i for i in range(len(slist)) if f"{fstep_over}.h5" in slist[i])
        slist.pop(j) # remove the final time solution
    except:
        slist = None

    if slist:
        for k in range(len(slist)):
            slist[k] = sdir + slist[k]



        run(["tar", "czf", sdir + "timesol.tar.gz"] + slist)
        run(["rm"] + slist)
