
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pickle

home = os.getenv('HOME')

res_dir = home + "/bedonian1/sample_torch1d_base_post/"
# res_dir = home + "/bedonian1/sample_tps2d_base_post_ELEC/"
# res_dir = home + "/bedonian1/sample_tps2d_base_post_ELEC_2/"



fqoin = res_dir + '/qoi_list.pickle' 
fqoda = res_dir + '/qoi_samples.pickle' 
with open(fqoin, 'rb') as f:
    qoi_list = pickle.load(f)
with open(fqoda, 'rb') as f:
    qoi_samples = pickle.load(f)


breakpoint()