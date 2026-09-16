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
"""

def listdir_nopickle(path):
    return [f for f in os.listdir(path) if not f.endswith('.pickle')]
def listdir_nocrash(path):
    return [f for f in os.listdir(path) if 'crashed' not in f]



home = os.getenv('HOME')

# source_file = "heated_Ar_3Dtorch_loMach/data.pvtu"
# velname = 'velocity'

source_file = "fullTorch_cold_Field_Sigfried/data.pvtu"
velname = 'vel'

res_file = "inlet_axi2d_interp_down1cm_heat.csv"
# res_file = "inlet_axi2d_interp_down05cm_heat.csv"
# res_file = "inlet_axi2d_interp_cold.csv"

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
# inlet_y = 0.0145
# inlet_y = 0.0145 + 0.005
inlet_y = 0.0145 + 0.01


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

data_arrays = {
    "temperature": np.zeros([n_rad, n_avg]),
    "velocity_X": np.zeros([n_rad, n_avg]),
    "velocity_Y": np.zeros([n_rad, n_avg]),
    "velocity_Z": np.zeros([n_rad, n_avg])
}

if 1:

    sol = pv.get_reader(source_file)
    # tf = sol.time_values[-1]
    # sol.set_active_time_value(tf)
    solg = sol.read()
    for s in range(n_avg):
    # for c_ind in cases[rank][:-2]:
    # for c_ind in [1]:

        angle = (s/n_avg)*2*np.pi

        print(f"Reading data at angle {angle} radians ...")

        inlet_coords = np.array([[0, inlet_y, 0],
                [inlet_r*np.cos(angle), inlet_y, inlet_r*np.sin(angle)]   ])

        # exit_line = pv.Spline(exit_coords, n_integ)
        # inlet_line = pv.Line(inlet_coords[0], inlet_coords[1], resolution=n_rad)

        # sold = solg.slice_along_line(inlet_line)
        sold = solg.sample_over_line(inlet_coords[0], inlet_coords[1], resolution=n_rad-1)

        ### radial and azimuthal conversion
        vrad = sold[velname][:,0]*np.cos(angle) + sold[velname][:,2]*np.sin(angle)
        vazm = -sold[velname][:,0]*np.sin(angle) + sold[velname][:,2]*np.cos(angle)

        if velname == "velocity":
            data_arrays["temperature"][:,s] = sold["temperature"]
        else:
            data_arrays["temperature"][:,s] = 300.

        data_arrays["velocity_X"][:,s] = vrad
        data_arrays["velocity_Y"][:,s] = sold[velname][:,1]
        data_arrays["velocity_Z"][:,s] = vazm

            

temp_avg = np.mean(data_arrays["temperature"], axis=1)
x_avg = np.mean(data_arrays["velocity_X"], axis=1)
y_avg = np.mean(data_arrays["velocity_Y"], axis=1)
z_avg = np.mean(data_arrays["velocity_Z"], axis=1)

temp_avg[-1] = 300.

# create inlet file
with open(res_file, 'w', newline='') as csvfile:
    inletwrite = csv.writer(csvfile)
    for i in range(n_rad):
        
        inletwrite.writerow([nat_ind[i], rad_avg[i], 0, 0, temp_avg[i], x_avg[i], y_avg[i], z_avg[i]])

breakpoint()