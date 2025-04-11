import numpy as np
import argparse
import os
from subprocess import run
import fileinput

"""
Generate 
"""

home = os.environ["HOME"]
base_script_path = "run_spoof_torch1dbig.sh"
sample_dir = f"{home}/bedonian1/test_spoof/"
script_dir = f"{home}/bedonian1/test_spoof/batch_scripts/"
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

    # open template

    sname = base_script.split('.')[0] + '_' + sample + '.sh'

    with open(base_script, 'r') as f:
        template = f.read()
    
    # replace sbatch rerun
    template = template.replace(base_script, script_dir + sname)

    # replace input_replace with path to sample director
    template = template.replace("%INPUT_REPLACE", sdir)

    # replace finish_replace with path to appropriate finish flag
    template = template.replace("%FINISH_REPLACE", sdir + '/finished')

    # write to script directory
    with open(script_dir + sname, 'w') as f:
        f.write(template)
    # with