import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py as h5
import scipy.constants as spc

from sample_utils import *
from sobol_tools import *
from pca_tools import *

"""
Script for correlation of torch1d outputs for the same rate under different model settings

Run after: 

resample_rates.py

**Run the torch1d samples for multiple meshes, models, etc. on the same rate samples**

sf_torch1d_uq_post.py
"""

home = os.getenv('HOME')
# in_dir = home + "/bedonian1/torch1d_resample_sens_r8/"

# include 4s when done
# out_dirs = [home + "/bedonian1/torch1d_post_r1_pilot_fine", home + "/bedonian1/torch1d_post_r1_pilot", home + "/bedonian1/torch1d_post_r1_pilot_coarse"]
# out_dirs = [home + "/bedonian1/tps2d_mf_post_r1/", home + "/bedonian1/torch1d_post_r1_pilot_fine", home + "/bedonian1/torch1d_post_r1_pilot", home + "/bedonian1/torch1d_post_r1_pilot_coarse", home + "/bedonian1/torch1d_post_r1_pilot_4s"]
# out_dirs = [home + "/bedonian1/tps2d_mf_post_r1_far/", home + "/bedonian1/torch1d_post_r1_pilot_fine", home + "/bedonian1/torch1d_post_r1_pilot", home + "/bedonian1/torch1d_post_r1_pilot_coarse", home + "/bedonian1/torch1d_post_r1_pilot_4s"]
out_dirs = [home + "/bedonian1/tps2d_mf_post_r1_far/", home + "/bedonian1/torch1d_post_r1_pilot_fine", home + "/bedonian1/torch1d_post_r1_pilot", home + "/bedonian1/torch1d_post_r1_pilot_coarse"]#, home + "/bedonian1/torch1d_post_r1_pilot_4s"]
# correspond to out_dirs order
# out_names = ["1D_Fine", "1D_Mid", "1D_Coarse"]
# out_names = ["1D_Fine", "1D_Mid", "1D_Coarse", "1D_4Species"]
# out_names = ["2D_Axi", "1D_Fine", "1D_Mid", "1D_Coarse", "1D_4Species"]
out_names = ["2D_Axi", "1D_Fine", "1D_Mid", "1D_Coarse"]
# max_sample = 0 # no limit
max_sample = 32

N_model = len(out_dirs)

# load outputs
qoi_val = {}
fqoin = out_dirs[0] + '/qoi_list.pickle' 
with open(fqoin, 'rb') as f:
    qoi_list = pickle.load(f)

for i, out_dir in enumerate(out_dirs):

    fqoiv = out_dir + f'/qoi_samples.pickle' 

    with open(fqoiv, 'rb') as f:
        qoi_val[out_names[i]] = pickle.load(f)
        
    if max_sample:
        for qoi in qoi_list:
            qoi_val[out_names[i]][qoi] = qoi_val[out_names[i]][qoi][:max_sample,:]

#Estimate Covariances

for qoi in qoi_list:
    
    Nout = qoi_val[out_names[0]][qoi].shape[1]
    # if qoi == 'exit_X':
    #     Nout = 1


    #concatenate results
    for j in range(Nout):
        qdata = np.zeros([N_model, qoi_val[out_names[0]][qoi].shape[0]])
        for i in range(N_model):
            qdata[i,:] = qoi_val[out_names[i]][qoi][:,j]

        # qcov = np.cov(qdata)
        qcorr = np.corrcoef(qdata)

        print(qoi + " " + str(j))
        print(qcorr)

    breakpoint()
    
    
breakpoint()
