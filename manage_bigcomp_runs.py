import numpy as np
import argparse
import os
from subprocess import run
import fileinput

"""
Generate 
"""

home = os.environ["HOME"]
base_script_path = "dane_scripts/run_tps2d_r1.sh"
sample_dir = f"{home}/bedonian1/tps2d_mf_r1_pilot_4/"
script_dir = f"{home}/bedonian1/tps2d_r1_batch_scripts_4/"

rfilename = 'tps_axi2d_input.ini'
rfilename2 = 'tps_axi2d_input_LT.ini'
rfilename0 = 'tps_axi2d_input_CT.ini'
prefix = "sig_A"

parser = argparse.ArgumentParser()
parser.add_argument('--run', action='store_true', default=False, help='flag to launch scripts')

args = parser.parse_args()
run_scripts = args.run

# if not run, then generate scripts
# parser.add_argument('--gen', help='flag to generate scripts')
base_script = base_script_path.split('/')[-1]

if run_scripts:

    sc_list = os.listdir(script_dir)
    
    for sc in sc_list:
        sc_file = script_dir + sc

        run(["sbatch", sc_file])

    quit()


# edit and copy the sbatch scripts

if not os.path.isdir(script_dir):
    os.makedirs(script_dir)

samples = os.listdir(sample_dir)

for sample in samples:

    if prefix not in sample:
        continue

    sdir = sample_dir + sample

    if rfilename is not None:
        # rfile = sdir + '/' + rfilename
        rfile = rfilename
    else:
        rfile = sdir
    # open template

    sname = base_script.split('.')[0] + '_' + sample + '.sh'

    with open(base_script_path, 'r') as f:
        template = f.read()
    
    # replace sbatch rerun
    template = template.replace(base_script, script_dir + sname)

    # replace input_replace with runfile option

    # change directory if we're using tps
    if rfilename is not None:
        # rfile = sdir + '/' + rfilename
        template = template.replace("%CDIR_REPLACE", 'cd ' + sdir)
    else:
        template = template.replace("%CDIR_REPLACE", '')
        
    template = template.replace("%INPUT_REPLACE", rfilename)

    # replace finish_replace with path to appropriate finish flag
    template = template.replace("%FINISH_REPLACE", sdir + '/step_up_time')

    # replace file switchers for next time step
    template = template.replace("%MV_REPLACE_1", f'mv {rfilename} {rfilename0}')
    template = template.replace("%MV_REPLACE_2", f'mv {rfilename2} {rfilename}')

    # write to script directory
    with open(script_dir + sname, 'w') as f:
        f.write(template)
    # with