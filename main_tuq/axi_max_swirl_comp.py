import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.pylab as plt
import vtk
import pyvista as pv
from mpi4py import MPI
import os, sys
import configparser
import shutil
import h5py as h5
import meshio
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
Obtain an axisymmetric initial condition from Sigfried's hot 3D simulation
by taking an azimuthal average
"""

def listdir_nopickle(path):
    return [f for f in os.listdir(path) if not f.endswith('.pickle')]
def listdir_nocrash(path):
    return [f for f in os.listdir(path) if 'crashed' not in f]

plt.rcParams['font.size'] = 15

home = os.getenv('HOME')

# will prepend max_ and avg_ to each plot
# plotname = 'swirl_comp_cold_newdomain.png'

## cold settings
# hot = False
# plotname = 'swirl_comp_cold_zetaf_complete.png'
# cold_3d_file = home + "/bedonian1/fullTorch_cold_Field_Sigfried/data.pvtu" # 3D solution
# cold_3d_data = "cold_3d_data.npy"
# cold_3d_data_avg = "cold_3d_data_avg.npy"

# cold_2d_file = [home + "/bedonian1/mean_tps2d_cold/output-cold-3/output-torch.pvd"]
# cold_2d_file = [home + "/bedonian1/mean_tps2d_CONDP3_TURB/output-cond-3/output-torch.pvd"]
# cold_2d_file = [home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_cold_down1cm_turb/output-torch/output-torch.pvd",
#                 # home + "/bedonian1/mean_tps2d_cold_21turb/output-cold-21turb/output-torch.pvd",
#                 # home + "/bedonian1/mean_tps2d_cold_16turb/output-torch/output-torch.pvd",
#                 # home + "/bedonian1/mean_tps2d_cold_11turb/output-cold-11turb/output-torch.pvd",
#                 # home + "/bedonian1/mean_tps2d_cold_t005turb/output-torch/output-torch.pvd",
#                 # home + "/bedonian1/mean_tps2d_cold_t0025turb/output-torch/output-torch.pvd",
#                 # home + "/bedonian1/mean_tps2d_cold_t0005turb/output-torch/output-torch.pvd",
#                 # home + "/bedonian1/mean_tps2d_cold_down1cm_21turb/output-torch/output-torch.pvd",
#                 # home + "/bedonian1/mean_tps2d_cold_down1cm_11turb/output-torch/output-torch.pvd",
#                 # home + "/bedonian1/mean_tps2d_cold_down1cm_t001turb/output-torch/output-torch.pvd",
#                 home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_cold_down1cm_t0001turb/output-torch/output-torch.pvd",
#                 home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_cold_down1cm_noturb/output-torch/output-torch.pvd",
#                 # home + "/bedonian1/mean_tps2d_cold_down1cm_zetaf/output-torch-zeta-f-FIX/output-torch-zeta-f.pvd",
#                 # home + "/bedonian1/mean_tps2d_cold_down1cm_zetaf/output-torch-zeta-f10/output-torch-zeta-f10.pvd",
#                 # home + "/bedonian1/mean_tps2d_cold_down1cm_zetaf/output-torch-zeta-f-100/output-torch-zeta-f-100.pvd",
#                 # home + "/bedonian1/mean_tps2d_cold_down1cm_zetaf/output-torch-zeta-f-10000/output-torch-zeta-f-10000.pvd"
#                 home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/output-torch-cold-v2-1/output-torch-cold-v2.pvd",
#                 ]

##hot settings
hot = True
plotname = 'swirl_comp_hot_zetaf_complete.png'
cold_3d_file = "/usr/workspace/utaustin/DROPBOX/heated_Ar_3Dtorch_loMach/data.pvtu" # 3D solution
cold_3d_data = "hot_3d_data.npy"
cold_3d_data_avg = "hot_3d_data_avg.npy"
cold_2d_file = [home + "/bedonian1/mean_tps2d_oldmesh/mean_tps2d_CONDP3_LTE/output-lte-2/output-torch.pvd",
                home + "/bedonian1/mean_tps2d_oldmesh/mean_tps2d_hot_t0001turb/output-hot-t0001turb/output-torch.pvd",
                # home + "/bedonian1/mean_tps2d_oldmesh/mean_tps2d_CONDP3_LTE_NT/output-torch/output-torch.pvd",
                # home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/output-torch-kappa-1/output-torch-kappa.pvd",
                home + "/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/output-torch-qt-2/output-torch-qt.pvd",
                ]



# name_map = [r'$\kappa = 0.41, l_{\text{max}} = 0.01$',
#             r'$\kappa = 0.21$',
#             # r'$\kappa = 0.16$',
#             r'$\kappa = 0.11$',
#             # r'$l_{\text{max}} = 0.005$',
#             # r'$l_{\text{max}} = 0.0025$',
#             # r'$l_{\text{max}} = 0.0005$',
#             r'$l_{\text{max}} = 0.001$',
#             r'$l_{\text{max}} = 0.0001$',
#             # r'$10\times$ fixed',
#             r'No turb, Down 1cm',
#             r'Zeta F, Down 1cm',
#             r'Zeta F 10',
#             r'Zeta F 100',
#             r'Zeta F 10000',
# ]
name_map = [r'ARANS',
            r'ARANS, $l_{\text{max}} = 10^{-4}$',
            # r'No turb',
            r'Zeta f'
]
# name_map = [r'2D max swirl']

# color_map = ['k', 'r', 'r', 'r', 'b', 'b', 'b']
# color_map = ['k', 'r', 'r', 'b', 'b', 'y', 'c', 'c', 'c', 'c']
# color_map = ['k', 'r', 'r', 'b', 'b']
# color_map = ['k', 'r', 'b', 'b']
# color_map = ['k', 'm', 'y', 'c']
color_map = ['k', 'm', 'c']
# color_map = ['k']
# alpha_map = [1.0, 1.0, 0.75, 0.5, 1.0, 0.75, 0.5]
# alpha_map = [1.0, 1.0, 0.75, 1.0, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
# alpha_map = [1.0, 1.0, 0.4, 1.0, 0.4]
# alpha_map = [1.0, 1.0, 1.0, 0.4]
# alpha_map = [1.0, 1.0, 1.0, 1.0]
# alpha_map = [1.0, 1.0, 1.0, 1.0]
alpha_map = [1.0, 1.0, 1.0]
# alpha_map = [1.0]


inlet_r = 0.028040
inlet_y = 0.0245
axi_l = 0.34

n_axi = 1000 # axial sampling (x-axis)
n_rad = 1000 # radial sampling for average or maximum (y-axis)
n_avg = 6 # azimuthal average of 3D solution

##########################################################################################################
# Script Starts Here
##########################################################################################################

if not os.path.isfile(cold_3d_data) or not os.path.isfile(cold_3d_data_avg):
    sol3 = pv.get_reader(cold_3d_file)
    # tf = sol.time_values[-1]
    # sol.set_active_time_value(tf)
    sol3g = sol3.read()
    max_swirl_3d = np.zeros(n_axi)
    avg_swirl_3d = np.zeros(n_axi)
    for i in range(n_axi):

        print(f"Point {i}")
        sws = np.zeros(n_rad)
        for s in range(n_avg):
        # for c_ind in cases[rank][:-2]:
        # for c_ind in [1]:

            angle = (s/n_avg)*2*np.pi

            print(f"Reading data at angle {angle} radians ...")

            rad_coords = np.array([[0, inlet_y+((i/n_axi)*axi_l), 0],
                    [inlet_r*np.cos(angle), inlet_y+((i/n_axi)*axi_l), inlet_r*np.sin(angle)]  ])
            
            sold = sol3g.sample_over_line(rad_coords[0], rad_coords[1], resolution=n_rad-1)
            if hot: # the hot dataset is labeled differently
                sws += -sold["velocity"][:,0]*np.sin(angle) + sold["velocity"][:,2]*np.cos(angle)
            else:
                sws += -sold["vel"][:,0]*np.sin(angle) + sold["vel"][:,2]*np.cos(angle)
            
        # rad_coords = np.array([[0, inlet_y+((i/n_axi)*axi_l), 0],
        #             [inlet_r, inlet_y+((i/n_axi)*axi_l), 0]  ])

        # sold = sol3g.sample_over_line(rad_coords[0], rad_coords[1], resolution=n_rad-1)
        # sws = sold["vel"][:,2]
        sws /= n_avg
        max_swirl_3d[i] = np.min(sws)
        avg_swirl_3d[i] = np.mean(sws)

    np.savetxt(cold_3d_data, max_swirl_3d)
    np.savetxt(cold_3d_data_avg, avg_swirl_3d)
else:
    max_swirl_3d = np.loadtxt(cold_3d_data)
    avg_swirl_3d = np.loadtxt(cold_3d_data_avg)


max_swirl_2d = []
avg_swirl_2d = []
for k, cfile in enumerate(cold_2d_file):
    sol2 = pv.get_reader(cfile)
    tf = sol2.time_values[-1]
    sol2.set_active_time_value(tf)
    sol2g = sol2.read()['Block-00']
    max_swirl_2d.append(np.zeros(n_axi))
    avg_swirl_2d.append(np.zeros(n_axi))
    for i in range(n_axi):

        print(f"Point {i}")
        rad_coords = np.array([[0, (i/n_axi)*axi_l, 0],
                    [inlet_r, (i/n_axi)*axi_l, 0]  ])

        sold = sol2g.sample_over_line(rad_coords[0], rad_coords[1], resolution=n_rad-1)
        sws = sold["swirl"][:]
        max_swirl_2d[k][i] = np.min(sws)
        avg_swirl_2d[k][i] = np.mean(sws)


# max plot
fig = plt.gcf()
fig.set_size_inches(8, 4.8)

plt.plot((np.arange(0,n_axi)/n_axi)*axi_l, abs(max_swirl_3d), 'k--', linewidth=1.8, label = '3D')
for k, cfile in enumerate(cold_2d_file):
    plt.plot((np.arange(0,n_axi)/n_axi)*axi_l, abs(max_swirl_2d[k]), color=color_map[k], alpha=alpha_map[k], label = name_map[k])

plt.xlabel("Axial Length (m)")
plt.ylabel("Max Swirl (m/s)")
plt.legend(fontsize=11, loc='upper right')
plt.ylim([-0.1, 28.2])
plt.grid()
plt.savefig("max_" + plotname, bbox_inches='tight', dpi = 600)
plt.clf()


#avg plot
fig = plt.gcf()
fig.set_size_inches(8, 4.8)

plt.plot((np.arange(0,n_axi)/n_axi)*axi_l, abs(avg_swirl_3d), 'k--', linewidth=1.8, label = '3D')
for k, cfile in enumerate(cold_2d_file):
    plt.plot((np.arange(0,n_axi)/n_axi)*axi_l, abs(avg_swirl_2d[k]), color=color_map[k], alpha=alpha_map[k], label = name_map[k])

plt.xlabel("Axial Length (m)")
plt.ylabel("Mean Swirl (m/s)")
plt.legend(fontsize=11, loc='upper right')
plt.ylim([-0.1, 10.0])
plt.grid()
plt.savefig("avg_" + plotname, bbox_inches='tight', dpi = 600)
plt.clf()


breakpoint()