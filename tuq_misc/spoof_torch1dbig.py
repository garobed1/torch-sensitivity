from tuq_util.file_util import t1dRestart
from subprocess import run
import sys, os

"""
Run a single torch1d case and handle restarts


"""

sdir = sys.argv[1]

### Options
# restart = True # enable restart from stopped state
pcomm = 'python3.11'
torch1d_infile = 'torch1d_input.yml'
fstep = 70000
run_torch1d = True        

# check if calculation already performed
if os.path.isdir(sdir + '/output/'):
    torch1d_infile_r, run_torch1d = t1dRestart(sdir, fstep, torch1d_infile)
else:
    torch1d_infile_r = torch1d_infile

# run
if run_torch1d:

    inputfile = sdir + '/' + torch1d_infile_r
    

    # run torch1d
    run([pcomm, "../torch1d.py", inputfile])