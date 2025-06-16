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

from tuq_util.sample_utils import *

# sys.path.insert(0, '/g/g14/bedonian1/torch1d/')
# from torch1d import *
# import inputs
# from axial_torch import AxialTorch

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
Script to perform post-processing of tps2d cases 

Includes area averaging of exit quantities
"""

def listdir_nopickle(path):
    return [f for f in os.listdir(path) if not f.endswith('.pickle')]
def listdir_nocrash(path):
    return [f for f in os.listdir(path) if 'crashed' not in f]



home = os.getenv('HOME')

# sample_in_dir = home + "/bedonian1/cross_section_samples_r7/"
# sample_out_dir = home + "/bedonian1/torch1d_samples_r7/"
# sample_out_dir = home + "/bedonian1/torch1d_resample_r7/"
# sample_out_dirs = [home + "/bedonian1/tps2d_mf_r1_pilot/",
#                    home + "/bedonian1/tps2d_mf_r1_pilot_2/",
#                    home + "/bedonian1/tps2d_mf_r1_pilot_3/",
#                    home + "/bedonian1/tps2d_mf_r1_pilot_4/",
#                    home + "/bedonian1/tps2d_mf_r1_pilot_5/"]
sample_out_dirs = [home + "/bedonian1/tps2d_mf_r1_G3/"]
template_file = f"{home}/bedonian1/mean_r6/r_lomach.torch.reacting.ini" # keep this to deal with restarts
infile_name = "/tps_axi2d_input.ini"
# res_dir = home + "/bedonian1/tps2d_mf_post_r1_far/"
# res_dir = home + "/bedonian1/tps2d_mf_post_r1_massflux_core/"
# res_dir = home + "/bedonian1/tps2d_mf_post_r1_wdata/"
res_dir = home + "/bedonian1/tps2d_mf_post_r1_G3/"

# number of integration points along axisymmetric line
# NOTE: not the actual number of points along the exterior slice,
# does nothing apparently
# time_avg = True
time_avg = False
time_sum = 15 # have time steps every 0.05 seconds, we would probably want to run with a configuration that outputs more frequently to do this


if time_avg:
    tf2 = time_sum
else:
    tf2 = 1

n_integ = 300

if len(sys.argv) > 2:
    res_dir = sys.argv[1]
    sample_out_dirs = list(sys.argv[2:])

# if len(sys.argv) > 3:
#     fstep = int(sys.argv[3])

# deal with heat dep later
# out_qoi = ['exit_p', 'exit_d', 'exit_v', 'exit_T', 'exit_X', 'heat_dep']
out_qoi = ['exit_p', 'exit_d', 'exit_v', 'exit_T', 'exit_X', 'exit_E']
# out_qoi = ['exit_E']

# soldata.array_names:
# ['CpMix', 'Qt', 'Rmix', 'Sjoule', 'Yn_Ar', 
# 'Yn_Ar.+1', 'Yn_Ar_h', 'Yn_Ar_m', 'Yn_Ar_p', 'Yn_Ar_r', 'Yn_E', 
# 'density', 'distance', 'emission', 'epsilon_rad', 'kappa', 'mu', 'muT', 
# 'pressure', 'resolution', 'sigma', 'sponge', 'swirl', 
# 'temperature', 'velocity', 'wall_dist', 'weff', 'attribute']
qoi_tps = ['temperature', 'pressure', 'velocity', 'density', 'Yn_Ar.+1',
            'Yn_Ar_m', 'Yn_Ar_r', 'Yn_Ar_p', 'Yn_Ar_h', 'Yn_E']
# get Yn_E to get electron density



make_plots = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 15,
})


# size = 3
# clist = os.listdir(sample_out_dir)
clist = []
for out_dir in sample_out_dirs:
    clist_c = listdir_nopickle(out_dir)
    for item in clist_c:
        clist.append(out_dir + item)
    
clist.sort()
# breakpoint()


N = len(clist)


cases = np.array_split(np.arange(N), size)

xdiff = 0.0145
# radius at exit
# exit_r = 0.01485
# exit_r = 0.025
exit_r = 0.0151
# exit_l = 0.355
# consider core to be at 0.14 on t1d, so 0.14 - 
exit_l = 0.3405
# exit_l = 0.14 - xdiff
# exit_l = 0.340
exit_coords = np.array([[0, exit_l, 0],
                [exit_r, exit_l, 0]   ])


qoi_sizes = {
    'exit_p': 1, 
    'exit_d': 2, 
    'exit_v': 1,
    'exit_T': 2,
    'exit_X': 5,
    'exit_E': 1,
    'heat_dep': 1
}

t1d_2_tps_names = {
    'exit_p': 'pressure',
    'exit_d': 'density',
    'exit_v': 'velocity',
    'exit_T': 'temperature',
    'exit_X': 'Yn_'
}

# get template adjustment from first case
c_inf = clist[0] + infile_name
args = configparser.ConfigParser()
args.optionxform = str
with open(c_inf, 'r') as f:
    args.read_file(f)


out_prefix = str(args['io']['outdirBase'])
if qoi_sizes['exit_X'] != int(args['plasma_models']['species_number']) - 2:
    qoi_sizes['exit_X'] = int(args['plasma_models']['species_number']) - 2

qoi_val_r = {}
qoi_val = {}
mpi_sizes = {}
mpi_offsets = {}


for qoi in out_qoi:
    mpi_sizes[qoi] = [cases[x].shape[0]*qoi_sizes[qoi] for x in range(size)]
    # mpi_offsets[qoi] = [0] + [cases[x].shape[0]*qoi_sizes[qoi] for x in range(size-1)]
    mpi_offsets[qoi] = [0] + np.cumsum(mpi_sizes[qoi][:-1]).tolist()
    qoi_val_r[qoi] = np.zeros([len(cases[rank]), qoi_sizes[qoi]])


if rank == 0:
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)

# groups = ['A', 'B', 'AB']
# process cases
if 1:

    c = 0
    for c_ind in cases[rank]:
    # for c_ind in [40, 41, 42, 43, 44, 45]:

        c_dir = clist[c_ind]
        c_inf = c_dir + infile_name
        r_dir = res_dir + '/' + clist[c_ind].split('/')[-1]
        c_name = clist[c_ind].split('/')[-1]

        if make_plots and not os.path.isdir(r_dir):
            os.makedirs(r_dir)

        if make_plots and not os.path.isdir(r_dir + '/plots'):
            os.makedirs(r_dir + '/plots')

        args = configparser.ConfigParser()
        args.optionxform = str
        with open(c_inf, 'r') as f:
            args.read_file(f)

        print(f"Reading {c_name} data ...")

        o_file = c_dir + '/' + out_prefix + '/' + out_prefix + '.pvd'
        sol = pv.get_reader(o_file)
        
        tf = sol.time_values[-1]
        print(f"Final time: {tf}")

        X_ion_t = []
        # for x, t in enumerate(sol.time_values):
        # breakpoint()
        # for x, t in enumerate(sol.time_values[-1:]):
        qdat = {}
        for qoi in out_qoi:
            qdat[qoi] = {}
            for k in range(qoi_sizes[qoi]):
                qdat[qoi][k] = []

        for x, t in enumerate(sol.time_values[-tf2:]):
            sol.set_active_time_value(t)
            solg = sol.read()

            # exit_line = pv.Spline(exit_coords, n_integ)
            exit_line = pv.Line(exit_coords[0], exit_coords[1], n_integ)
            sold = solg.slice_along_line(exit_line)
            soldata = sold['Block-00']

            qoi_data_r = {}
            for qoi in out_qoi:
                if qoi == "exit_T": # same temp for both Ar and E
                    qoi_data_r[qoi] = [soldata['temperature'], soldata['temperature']]
                if qoi == "exit_p":
                    qoi_data_r[qoi] = [soldata['pressure']]
                if qoi == "exit_v": # axial velocity only
                    qoi_data_r[qoi] = [soldata['velocity'][:,1]]
                if qoi == "exit_d": # axial velocity only
                    qoi_data_r[qoi] = [soldata['density']*soldata['Yn_Ar'], soldata['density']*soldata['Yn_E']*1e5]
                if qoi == "exit_X":
                    qoi_data_r[qoi] = []
                    qoi_data_r[qoi].append(soldata['Yn_Ar.+1'])
                    if qoi_sizes["exit_X"] == 5:
                        qoi_data_r[qoi].append(soldata['Yn_Ar_m'])
                        qoi_data_r[qoi].append(soldata['Yn_Ar_r'])
                        qoi_data_r[qoi].append(soldata['Yn_Ar_p'])
                        qoi_data_r[qoi].append(soldata['Yn_Ar_h'])
                    elif qoi_sizes["exit_X"] == 2:
                        qoi_data_r[qoi].append(soldata['Yn_Ar_s'])
                    else:
                        print("Invalid number of species")
                        quit()
                if qoi == "exit_E":
                    CP = float(args['species/species1']['perfect_mixture/constant_molar_cp'])
                    qoi_data_r[qoi] = [soldata['temperature']*spc.R*CP/0.039948] # /argon molar density


            # get mask for the points actually within the extent of the outlet
            mask = [0] + [x for x in range(1, soldata.points.shape[0]) if soldata.points[x-1,0] < exit_r]

            # precompute denominator
            # int_denom = np.pi*exit_r*exit_r

            # mass-weighted average
            int_denom = 0
            for i in range(len(mask) - 1):
                work = (soldata['velocity'][i,1]*soldata['density'][i] + 
                        soldata['velocity'][i+1,1]*soldata['density'][i+1])/2.
                work *= (soldata.points[i,0] + soldata.points[i+1,0])/2.
                work *= (soldata.points[i+1,0] - soldata.points[i,0])
                int_denom += work
            int_denom *= 2*np.pi

            # now area average integrate all qoi
            for qoi in out_qoi:
                for k in range(len(qoi_data_r[qoi])):
                    
                    # area average
                    # isum = 0
                    # for i in range(len(mask) - 1):
                    #     # 2 pi r h(r) dr rings by trapezoid
                    #     work = (qoi_data_r[qoi][k][i] + qoi_data_r[qoi][k][i+1])/2.
                    #     work *= (soldata.points[i,0] + soldata.points[i+1,0])/2.
                    #     # work = (qoi_data_r[qoi][k][i]*soldata.points[i,0] + qoi_data_r[qoi][k][i+1]*soldata.points[i+1,0])/2.
                    #     work *= (soldata.points[i+1,0] - soldata.points[i,0])
                    #     isum += work

                    # # res = np.trapz(qoi_data_r[qoi][k][mask], x=soldata.points[mask,0])
                    # work2 = (isum*2*np.pi)/int_denom
                    # qoi_val_r[qoi][c,k] = work2

                    # mass-flux average??
                    isum = 0
                    for i in range(len(mask) - 1):
                        # 2 pi r h(r) dr rings by trapezoid
                        # work = (qoi_data_r[qoi][k][i]*soldata['velocity'][i,1]*soldata['density'][i] + 
                        #         qoi_data_r[qoi][k][i+1]*soldata['velocity'][i+1,1]*soldata['density'][i+1])/2.
                        # work *= (soldata.points[i,0] + soldata.points[i+1,0])/2.
                        work = (qoi_data_r[qoi][k][i]*soldata.points[i,0]*soldata['velocity'][i,1]*soldata['density'][i] + 
                                qoi_data_r[qoi][k][i+1]*soldata.points[i+1,0]*soldata['velocity'][i+1,1]*soldata['density'][i+1])/2.
                        work *= (soldata.points[i+1,0] - soldata.points[i,0])
                        isum += work

                    # res = np.trapz(qoi_data_r[qoi][k][mask], x=soldata.points[mask,0])
                    int_numer = isum*2*np.pi
                    work2 = int_numer/int_denom
                    qdat[qoi][k].append(work2)

        for qoi in out_qoi:
            for k in range(qoi_sizes[qoi]):
                qoi_val_r[qoi][c,k] = np.mean(qdat[qoi][k])
                # breakpoint()
            # X_ion_t.append(qoi_val_r['exit_X'][c,0])
        
        # plt.plot(sol.time_values, X_ion_t)
        # plt.savefig("xion_over_time_tps.png")
        # plt.clf()
        # breakpoint()
                # print(int_numer)
                # print(work2)
        # breakpoint()


        c += 1


    for qoi in out_qoi:
        qoi_val[qoi] = np.zeros([N, qoi_sizes[qoi]])
        # breakpoint()
        comm.Gatherv(qoi_val_r[qoi], [qoi_val[qoi], mpi_sizes[qoi], mpi_offsets[qoi], MPI.DOUBLE],  root=0)
        # comm.Gatherv(qoi_val_r[group][qoi], [qoi_val[group][qoi], mpi_sizes[group], mpi_offsets],  root=0)

    import pickle
    # if sobol_samples:
    #     filename = res_dir + f'/qoi_samples_{group}.pickle' 
    # else:
    filename = res_dir + '/qoi_samples.pickle' 

    if rank == 0:
        with open(filename, 'wb') as f:

            pickle.dump(qoi_val, f)

        with open(res_dir + '/qoi_list.pickle', 'wb') as f:

            pickle.dump(out_qoi, f)

    # breakpoint()