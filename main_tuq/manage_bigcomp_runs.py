import numpy as np
import argparse
import os
from subprocess import run
import fileinput

"""
Generate and run scripts to run samples of TPS on the cluster


"""


home = os.environ["HOME"]



### TPS (2D) Sample Directories
# sample_dir = f"{home}/bedonian1/tps2d_mf_r1_pilot_5/"
# sample_dir = f"{home}/bedonian1/tps2d_mf_r1_pilot_4s_1/"
# sample_dir = f"{home}/bedonian1/tps2d_mf_r1_pilot_LF_1/"
# sample_dir = f"{home}/bedonian1/tps2d_mf_r1_pilot_LF2_1_T2/"
# sample_dir = f"{home}/bedonian1/tps2d_time_test_2/"
# sample_dir = f"{home}/bedonian1/tps2d_mf_r1_G4/"
sample_dir = f"{home}/bedonian1/tps2d_mf_r1_G3/"

### Template for Job Submission Script
# base_script_path = "dane_scripts/run_tps2d_r1.sh"
# base_script_path = "dane_scripts/run_tps2d_stage.sh"
# base_script_path = "dane_scripts/run_tps2d_LF2_stage.sh"
base_script_path = "dane_scripts/run_tps2d_stage.sh"

### Job Submission Script Location
# script_dir = f"{home}/bedonian1/tps2d_r1_batch_scripts_5/"
# script_dir = f"{home}/bedonian1/tps2d_r1_batch_scripts_4s_1/"
# script_dir = f"{home}/bedonian1/tps2d_r1_batch_scripts_LF2_1_T2/"
# script_dir = f"{home}/bedonian1/tps2d_time_test_scripts_2/"
script_dir = f"{home}/bedonian1/tps2d_r1_batch_scripts_G3/"


### Names of Successive Input Files per Sample
rfilename = 'tps_axi2d_input.ini'
rfilenames = ['tps_axi2d_input_0.ini', 'tps_axi2d_input_1.ini', 'tps_axi2d_input_2.ini', 'tps_axi2d_input_3.ini', 'tps_axi2d_input_4.ini']
# rfilename = 'tps_axi2d_input.ini'
# rfilename2 = 'tps_axi2d_input_LT.ini'
# rfilename0 = 'tps_axi2d_input_CT.ini'
prefix = "sig_A"




##########################################################################################################
# Script Starts Here
##########################################################################################################



parser = argparse.ArgumentParser()
parser.add_argument('--run', action='store_true', default=False, help='flag to launch scripts')
parser.add_argument('--r1', action='store', default=0, help='case start')
parser.add_argument('--r2', action='store', default=0, help='case end')

args = parser.parse_args()
run_scripts = args.run
r1 = int(args.r1)
r2 = int(args.r2)

# if not run, then generate scripts
# parser.add_argument('--gen', help='flag to generate scripts')
base_script = base_script_path.split('/')[-1]

if run_scripts:

    sc_list = os.listdir(script_dir)
    sc_list.sort()    

    for i in range(r1, min(r2, len(sc_list))):
    # for sc in sc_list:
        sc_file = script_dir + sc_list[i]

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

    with open(base_script_path, 'r') as f:
        template = f.read()
    
    # replace sbatch rerun
    template = template.replace(base_script, script_dir + sname)

    # replace input_replace with runfile option

    if rfilenames is not None:
        # rfile = sdir + "/" + rfilename
        rfile = rfilenames[0]
    else:
        rfile = sdir

    # change directory if we're using tps
    if rfilenames is not None:
        # rfile = sdir + "/" + rfilename
        template = template.replace("%CDIR_REPLACE", 'cd ' + sdir)
    else:
        template = template.replace("%CDIR_REPLACE", '')
        
    template = template.replace("%INPUT_REPLACE", sdir + "/" + rfilename)

    # replace finish_replace with path to appropriate finish flag
    template = template.replace("%FINISH_REPLACE", sdir + '/finished')

    # replace file switchers for next time step
    template = template.replace("%MV_REPLACE_0", f'mv {sdir + "/" + rfilenames[0]} {sdir + "/" + rfilename}')
    for i in range(1, len(rfilenames)):
        template = template.replace(f"%MV_REPLACE_{i}", f'mv {sdir + "/" + rfilenames[i]} {sdir + "/" + rfilenames[i-1]}')

    template = template.replace("%Q_REPLACE", f'{sdir + "/" + rfilenames[0]}')

    # write to script directory
    with open(script_dir + sname, 'w') as f:
        f.write(template)
    # with