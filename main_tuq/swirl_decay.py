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
Plot the mass-averaged swirl in the axial direction over time
Axisymmetric cases, and
Sigfried's 3D cases
"""

def listdir_nopickle(path):
    return [f for f in os.listdir(path) if not f.endswith('.pickle')]
def listdir_nocrash(path):
    return [f for f in os.listdir(path) if 'crashed' not in f]


plot_name = "swirl_decay_11turb.png"

home = os.getenv('HOME')

source_dirs = [
    home + "/bedonian1/mean_tps2d_INLETP3_NOTURB/output-hp3-noturb-1/",
    home + "/bedonian1/mean_tps2d_INLETP3_TURB/output-hp3-turb-1/",
    home + "/bedonian1/mean_tps2d_INLETP3_FIXJOULe/output-hp3-fixjoule-1/"
]
source_names = [
    "no turb",
    "turb",
    "fix joule"
]
source_color = [
    "k",
    "r",
    "g"
]
source_3d = "/usr/workspace/utaustin/DROPBOX/heated_Ar_3Dtorch_loMach/data.pvtu"
# res_file = "inlet_axi2d_interp_fullheat.csv"

n_rad = 200 # number of integration points
n_avg = 150 # number of axial points to sample

length = 0.34
inlet_r = 0.028040
inlet_y = 0.0145 # for 3D only


##########################################################################################################
# Script Starts Here
##########################################################################################################

### Radii
axial_avg = np.linspace(0, length, n_avg)

base_coords = np.array([[0, 0, 0],
                [inlet_r, 0, 0]   ])

data_arrays = {}
mass_avgs = {}

for i, src_dir in enumerate(source_dirs):

    
    sol = pv.get_reader(src_dir+'/output-torch.pvd')
    name = source_names[i]
    # tf = sol.time_values[-1]
    # sol.set_active_time_value(tf)
    times = sol.time_values[-6:]
    nt = len(times)

    data_arrays[i] = {"swirl": np.zeros([nt, n_avg, n_rad])}
    mass_avgs[i] = {"swirl": np.zeros([nt, n_avg])}
    print(f"Dataset {source_names[i]}  ...")
    for ts, t in enumerate(times):
        print(f"Time {t} ...")
        sol.set_active_time_value(t)
        solg = sol.read()
        for s in range(n_avg):
        # for c_ind in cases[rank][:-2]:
        # for c_ind in [1]:
            

            # angle = (s/n_avg)*2*np.pi
            dist = axial_avg[s]

            print(f"Integrating swirl data at {dist} meters ...")

            inlet_coords = np.array([[0, dist, 0],
                    [inlet_r, dist, 0]   ])
                
            # use for the 3D
            # inlet_coords = np.array([[0, inlet_y, 0],
            #         [inlet_r*np.cos(angle), inlet_y, inlet_r*np.sin(angle)]   ])

            # exit_line = pv.Spline(exit_coords, n_integ)
            # inlet_line = pv.Line(inlet_coords[0], inlet_coords[1], resolution=n_rad)

            # sold = solg.slice_along_line(inlet_line)
            solb = solg['Block-00']
            sold = solb.sample_over_line(inlet_coords[0], inlet_coords[1], resolution=n_rad-1)
            data_arrays[i]["swirl"][ts,s,:] = sold["swirl"]

            # integrate
            mask = [0] + [x for x in range(1, sold.points.shape[0]) if not np.isnan(sold["swirl"][x])]
            isum = 0
            for j in range(len(mask) - 1):
                work = (sold["swirl"][j]*sold.points[j,0]*sold['density'][j] + 
                        sold["swirl"][j+1]*sold.points[j+1,0]*sold['density'][j+1])/2.
                work *= (sold.points[j+1,0] - sold.points[j,0])
                isum += work
            work2 = isum*2*np.pi
            mass_avgs[i]["swirl"][ts,s] = work2


T_t = []
            
for i in range(len(source_dirs)):        
    for ts in range(data_arrays[i]["swirl"].shape[0]): 
        tt = data_arrays[i]["swirl"].shape[0]
        if ts == tt - 1:
            plt.plot(axial_avg, mass_avgs[i]["swirl"][ts,:], color = source_color[i], label = source_names[i])
        else:
            plt.plot(axial_avg, mass_avgs[i]["swirl"][ts,:], color = source_color[i], alpha = (1.0 - ((tt-ts-1)/tt)*0.7))
# plt.ylim(-0.003, 0.0)
plt.xlabel(r"$y$ (m)")
plt.ylabel(r"$\dot{V}_\theta (y)$")
plt.legend()
plt.savefig(plot_name, dpi = 600)
plt.clf()
breakpoint()