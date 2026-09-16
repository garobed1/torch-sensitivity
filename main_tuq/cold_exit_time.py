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
Plot exit properties of cold flow over time
"""

def listdir_nopickle(path):
    return [f for f in os.listdir(path) if not f.endswith('.pickle')]
def listdir_nocrash(path):
    return [f for f in os.listdir(path) if 'crashed' not in f]


plot_name = "vrm12_CFL01_avg_yvel_exit_cold_zetaf.png"
plot_name_2 = "vrm12_CFL01_max_yvel_exit_cold_zetaf.png"
# plot_name = "vrm12_avg_yvel_exit_cold_zetaf.png"
# plot_name_2 = "vrm12_max_yvel_exit_cold_zetaf.png"
# plot_name = "urm11_avg_yvel_exit_cold_zetaf.png"
# plot_name_2 = "urm11_max_yvel_exit_cold_zetaf.png"
# plot_name = "rm7_avg_yvel_exit_cold_zetaf.png"
# plot_name_2 = "rm7_max_yvel_exit_cold_zetaf.png"

home = os.getenv('HOME')

source_dirs = [
    home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/output-torch-cold-v2-rm12-2-1/output-torch-cold-v2-rm12.pvd",
    # home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/output-torch-cold-v2-rm12/output-torch-cold-v2-rm12.pvd",
    # home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/output-torch-cold-v2-rm12-2/output-torch-cold-v2-rm12.pvd",
    # home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/output-torch-cold-w2-rm11-1/output-torch-cold-w2-rm11.pvd",
    # home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/output-torch-cold-w2-rm9-1/output-torch-cold-w2-rm9.pvd",
    # home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/output-torch-cold-w2-rm9-2/output-torch-cold-w2-rm9.pvd",
    # home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/output-torch-cold-w2-rm9-3/output-torch-cold-w2-rm9.pvd",
    # home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/output-torch-cold-v2-1/output-torch-cold-v2.pvd",
    # home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/output-torch-cold-v2-2/output-torch-cold-v2.pvd",
    # home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/output-torch-cold-v2-3/output-torch-cold-v2.pvd",
]
# source_dirs = [
#     home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/output-torch-cold/output-torch-cold.pvd",
# ]


n_rad = 200 # number of integration points
# n_avg = 150 # number of axial points to sample

entry = 0.01
length = 0.34
inlet_r = 0.028040
#exit_r = 0.0145 # for 3D only
exit_r = 0.015 # for 3D only
density = 1.603

##########################################################################################################
# Script Starts Here
##########################################################################################################

### Radii
# axial_avg = np.linspace(0, length, n_avg)

# base_coords = np.array([[0, 0, 0],
#                 [inlet_r, 0, 0]   ])

data_arrays = {}
mass_avgs = {}
entry_mass_avgs = {}
center = {}
maxval = {}
times = []
for i, src_dir in enumerate(source_dirs):

    
    sol = pv.get_reader(src_dir)
    # name = source_names[i]
    # tf = sol.time_values[-1]
    # sol.set_active_time_value(tf)
    times.append(sol.time_values[:])
    nt = len(times[i])

    data_arrays[i] = {"velocity_Y": np.zeros([nt, n_rad])}
    mass_avgs[i] = {"velocity_Y": np.zeros([nt])}
    entry_mass_avgs[i] = {"velocity_Y": np.zeros([nt])}
    center[i] = {"velocity_Y": np.zeros([nt])}
    maxval[i] = {"velocity_Y": np.zeros([nt])}
    print(f"Dataset {i}  ...")
    for ts, t in enumerate(times[i]):
        print(f"Time {t} ...")
        sol.set_active_time_value(t)
        solg = sol.read()
        # for s in range(n_avg):
        # for c_ind in cases[rank][:-2]:
        # for c_ind in [1]:
            

        # angle = (s/n_avg)*2*np.pi
        # dist = axial_avg[s]

        print(f"Integrating velocity data at time {t} ...")

        # inlet_coords = np.array([[0, dist, 0],
        #         [exit_r, dist, 0]   ])
        inlet_coords = np.array([[0, length, 0],
                [exit_r, length, 0]   ])
        
        inlet_coords_2 = np.array([[0, entry, 0],
                [inlet_r, entry, 0]   ])
            
        # use for the 3D
        # inlet_coords = np.array([[0, inlet_y, 0],
        #         [inlet_r*np.cos(angle), inlet_y, inlet_r*np.sin(angle)]   ])

        # exit_line = pv.Spline(exit_coords, n_integ)
        # inlet_line = pv.Line(inlet_coords[0], inlet_coords[1], resolution=n_rad)

        # sold = solg.slice_along_line(inlet_line)
        solb = solg['Block-00']
        sold = solb.sample_over_line(inlet_coords[0], inlet_coords[1], resolution=n_rad-1)
        data_arrays[i]["velocity_Y"][ts,:] = sold["velocity"][:,1]
        
        sold2 = solb.sample_over_line(inlet_coords_2[0], inlet_coords_2[1], resolution=n_rad-1)

        # integrate
        mask = [0] + [x for x in range(1, sold.points.shape[0]) if not np.isnan(sold["velocity"][x,1])]
        isum = 0
        for j in range(len(mask) - 1):
            work = (sold["velocity"][j,1]*sold.points[j,0]*density + #sold['density'][j] + 
                    sold["velocity"][j+1,1]*sold.points[j+1,0]*density)/2. #sold['density'][j+1])/2.
            # breakpoint()
            work *= (sold.points[j+1,0] - sold.points[j,0])
            isum += work
        work2 = isum*2*np.pi
        mass_avgs[i]["velocity_Y"][ts] = work2
        center[i]["velocity_Y"][ts] = sold["velocity"][0,1]
        maxval[i]["velocity_Y"][ts] = max(sold["velocity"][:,1])
        
        mask = [0] + [x for x in range(1, sold2.points.shape[0]) if not np.isnan(sold2["velocity"][x,1])]
        isum = 0
        for j in range(len(mask) - 1):
            work = (sold2["velocity"][j,1]*sold2.points[j,0]*density + #sold['density'][j] + 
                    sold2["velocity"][j+1,1]*sold2.points[j+1,0]*density)/2. #sold['density'][j+1])/2.
            # breakpoint()
            work *= (sold2.points[j+1,0] - sold2.points[j,0])
            isum += work
        work2 = isum*2*np.pi
        entry_mass_avgs[i]["velocity_Y"][ts] = work2
        breakpoint()


T_t = []
            
for i in range(len(source_dirs)):        
    for ts in range(data_arrays[i]["velocity_Y"].shape[0]): 
        tt = data_arrays[i]["velocity_Y"].shape[0]
        # if ts == tt - 1:
        plt.plot(times[i], mass_avgs[i]["velocity_Y"], color = 'k', label='inlet')
        plt.plot(times[i], entry_mass_avgs[i]["velocity_Y"], color = 'b', label='exit')
        # else:
        #     plt.plot(axial_avg, mass_avgs[i]["swirl"][ts,:], color = source_color[i], alpha = (1.0 - ((tt-ts-1)/tt)*0.7))
# plt.ylim(-0.003, 0.0)
plt.xlabel(r"$y$ (m)")
plt.ylabel(r"$ (y)$")
plt.legend()
plt.savefig(plot_name, dpi = 600)
plt.clf()

for i in range(len(source_dirs)):        
    # for ts in range(data_arrays[i]["velocity_Y"].shape[0]): 
    tt = data_arrays[i]["velocity_Y"].shape[0]
    # if ts == tt - 1:
    plt.plot(times[i], maxval[i]["velocity_Y"], color = 'k')
    plt.plot(times[i], center[i]["velocity_Y"], color = 'b')

plt.plot([],[], color = 'k', label='max')
plt.plot([],[], color = 'b', label='center')
        # else:
        #     plt.plot(axial_avg, mass_avgs[i]["swirl"][ts,:], color = source_color[i], alpha = (1.0 - ((tt-ts-1)/tt)*0.7))
# plt.ylim(-0.003, 0.0)
plt.xlabel(r"$y$ (m)")
plt.ylabel(r"$ (y)$")
plt.legend()
plt.savefig(plot_name_2, dpi = 600)
plt.clf()
breakpoint()