import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.pylab as plt
import vtk
import pyvista as pv
from mpi4py import MPI
import os, sys
import configparser
import shutil
import scipy.constants as spc
import csv

# sys.path.insert(0, '/g/g14/bedonian1/torch1d/')
# from torch1d import *
# import inputs
# from axial_torch import AxialTorch

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
Axisymmetric average of inlet from Sigfried's 3D cases
Azimuthal average of inlet profile for turbulent quantities k and v2
"""

def listdir_nopickle(path):
    return [f for f in os.listdir(path) if not f.endswith('.pickle')]
def listdir_nocrash(path):
    return [f for f in os.listdir(path) if 'crashed' not in f]



home = os.getenv('HOME')

# source_file = "heated_Ar_3Dtorch_loMach/data.pvtu"
# velname = 'velocity'

source_file = home + "/meshes/fullTorch_cold_Field_Sigfried/data.pvtu"
velname = 'rms'

res_file = "tke_3d.csv"

### Number of radial points to take averages for
n_rad = 999
n_avg = 1000 # number of azimuthal points to sample

### Time Average (Don't use, not enough sampling)
time_avg = False
time_sum = 18 # have time steps every 0.05 seconds, we would probably want to run with a configuration that outputs more frequently to do this
if time_avg:
    tf2 = time_sum
else:
    tf2 = 1

inlet_r = 0.028040
# inlet_y = 0.0145 #original start point
inlet_y = 0.0245 #down1cm
inlet_y_2d = 0.01 # the inlet coordinate in the 2D mesh (down1cm)

##########################################################################################################
# Script Starts Here
##########################################################################################################

### Radii
rad_avg = np.linspace(0, inlet_r, n_rad)

base_coords = np.array([[0, inlet_y, 0],
                [inlet_r, inlet_y, 0]   ])

nat_ind = np.arange(1, n_rad+1)

# t1d_2_tps_names = {
#     'exit_p': 'pressure',
#     'exit_d': 'density',
#     'exit_v': 'velocity',
#     'exit_T': 'temperature',
#     'exit_X': 'Yn_'
# }

## rms
# 0. XX
# 1. YY
# 2. ZZ
# 3. XY
# 4. XZ
# 5. YZ

data_arrays = {
    "TKE": np.zeros([n_rad, n_avg]),
    "V2": np.zeros([n_rad, n_avg]),
}

if 1:

    sol = pv.get_reader(source_file)
    # tf = sol.time_values[-1]
    # sol.set_active_time_value(tf)
    solg = sol.read()
    for s in range(n_avg):

        angle = (s/n_avg)*2*np.pi

        print(f"Reading data at angle {angle} radians ...")

        inlet_coords = np.array([[0, inlet_y, 0],
                [inlet_r*np.cos(angle), inlet_y, inlet_r*np.sin(angle)]   ])

        sold = solg.sample_over_line(inlet_coords[0], inlet_coords[1], resolution=n_rad-1)

        ### turbulent kinetic energy
        tke_line = 0.5*(sold[velname][:,0] + sold[velname][:,1] + sold[velname][:,2])
        v2_line = sold[velname][:,0]*np.cos(angle) + sold[velname][:,2]*np.sin(angle)

        data_arrays["TKE"][:,s] = tke_line
        data_arrays["V2"][:,s] = v2_line

            
tke_avg = np.mean(data_arrays["TKE"], axis=1)
v2_avg = np.mean(data_arrays["V2"], axis=1)

# create inlet file
with open(res_file, 'w', newline='') as csvfile:
    inletwrite = csv.writer(csvfile)
    for i in range(n_rad):
        
        # tke.csv format for tps
        inletwrite.writerow([rad_avg[i], inlet_y_2d, 0, tke_avg[i], v2_avg[i]])

breakpoint()