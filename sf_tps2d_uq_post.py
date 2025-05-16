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

from sample_utils import *

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
sample_out_dirs = [home + "/bedonian1/tps2d_mf_r1_pilot/",
                   home + "/bedonian1/tps2d_mf_r1_pilot_2/",
                   home + "/bedonian1/tps2d_mf_r1_pilot_3/",
                   home + "/bedonian1/tps2d_mf_r1_pilot_4/",
                   home + "/bedonian1/tps2d_mf_r1_pilot_5/"]
template_file = f"{home}/bedonian1/mean_r6/r_lomach.torch.reacting.ini" # keep this to deal with restarts
infile_name = "/tps_axi2d_input.ini"
# res_dir = home + "/bedonian1/tps2d_mf_post_r1_far/"
# res_dir = home + "/bedonian1/tps2d_mf_post_r1_massflux_core/"
res_dir = home + "/bedonian1/tps2d_mf_post_r1_time_avg/"

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



    #     # set up residual, and get "final" time step
    #     from os.path import exists
    #     rf = None
    #     # Nr = divmod(solver.timeIntegrator.Nt, solver.outputFrequency)[0]
    #     Nr = len(listdir_nocrash(solver.outputDir)) - 1
    #     filenames = []
    #     for r in range(Nr+1):
    #         filenames += ['%s/%s-%08d.h5' % (solver.outputDir, solver.prefix, nt0 + r * solver.outputFrequency)]
            
    #         if nt0 + r * solver.outputFrequency == fstep and exists(filenames[-1]): 
    #             rf = r + 0
    #     # breakpoint()
    #     crashFile = '%s/%s.crashed.h5' % (solver.outputDir, solver.prefix)
    #     if (exists(crashFile)):
    #         filenames += [crashFile]
    #         Nr += 1

    #     if rf is None and skip_incomplete:
    #         continue
    #     elif rf is None and not skip_incomplete:
    #         rf = -1

    #     hist = np.zeros([Nr+1, solver.grid.Nx, solver.state.nUnknowns])
    #     dt = solver.timeIntegrator.dt

    #     area = np.pi * (solver.grid.radius ** 2)
    #     heatSwitch = config.getInput(['system','heat-source','enabled'], fallback=False)
    #     wallSwitch = config.getInput(['system','wall-contribution','enabled'], fallback=False)
    #     necSwitch = config.getInput(['system','net-emission','enabled'], fallback=False)
    #     numRxn = len(config.getInput(['reactions'], fallback=[]))
    #     chemSwitch = (numRxn > 0)
    #     rxnEqn = []
    #     if (heatSwitch):
    #         from axial_torch_rhs2 import AxialTorchHeatSource
    #         heatSrc = AxialTorchHeatSource(solver.state, config)
    #     if (wallSwitch):
    #         from axial_torch_rhs2 import AxialTorchWallContribution
    #         wallSrc = AxialTorchWallContribution(solver.state, config)
    #     if (necSwitch):
    #         from axial_torch_rhs2 import AxialTorchNetEmission
    #         necSrc = AxialTorchNetEmission(solver.state, config)
    #     if (chemSwitch):
    #         from reaction import MassActionLaw
    #         chem = MassActionLaw(solver.state, config)
    #         rxnList = config.getInput(['reactions'], fallback=[])
    #         for rxn in rxnList:
    #             rxnEqn += [rxn['equation']]

    #     from axial_torch_rhs import AxialTorchDivergence
    #     divRhs = AxialTorchDivergence(solver.state)

    #     # get states
    #     div = np.zeros([Nr+1, solver.grid.Nx])
    #     rho = np.zeros([Nr+1, solver.grid.Nx])
    #     Th = np.zeros([Nr+1, solver.grid.Nx])
    #     Te = np.zeros([Nr+1, solver.grid.Nx])
    #     vel = np.zeros([Nr+1, solver.grid.Nx, solver.state.nVel])
    #     nsp = np.zeros([Nr+1, solver.grid.Nx, solver.state.nSpecies])
    #     Xsp = np.zeros([Nr+1, solver.grid.Nx, solver.state.nSpecies])
    #     continuity = np.zeros([Nr+1, solver.grid.Nx])
    #     plasmaPower = np.zeros(Nr+1,)
    #     heat = np.zeros([Nr+1, solver.grid.Nx])
    #     wall = np.zeros([Nr+1, solver.grid.Nx])
    #     nec = np.zeros([Nr+1, solver.grid.Nx])
    #     if (chemSwitch):
    #         rxn = np.zeros([Nr+1, chem.nRxn, solver.grid.Nx])
    #         rxnb = np.zeros([Nr+1, chem.nRxn, solver.grid.Nx])
    #         ndots = np.zeros([Nr+1, solver.state.nSpecies, solver.grid.Nx])

    #     if (len(config.getInput(['reactions'], fallback=[])) > 0):
    #         from reaction import MassActionLaw
    #         chem = MassActionLaw(solver.state, config)
    #     for r, filename in enumerate(filenames):
    #     #     filename = '%s/%s-%08d.h5' % (solver.outputDir, solver.prefix, nt0 + r * solver.outputFrequency)
    #         solver.state.loadState(filename)
    #         # solver.rhs.collInt.update(solver.state)
    #         solver.state.collInt.update(solver.state)
    #     #     transport.update(solver.state)
    #         hist[r] = np.copy(solver.state.conserved)
    #     #     solver.state.conserved = np.copy(hist[r])
    #     #     solver.state.time = r * solver.outputFrequency * dt
    #     # #     print(solver.state.time)
    #     #     solver.state.update()
    #     #     rhs0 = solver.rhs.compute(solver.state)
    #         div[r] = -np.copy(divRhs.compute(solver.state)[:,0])
    #     #     rho[r] = np.copy(solver.state.density.var)
    #         rho[r] = np.copy(solver.state.conserved[:,1+nVel])
    #         vel[r] = np.copy(solver.state.velocity.var)
    #         Th[r] = np.copy(solver.state.Th.var)
    #         Te[r] = np.copy(solver.state.Te.var)
    #         nsp[r] = np.copy(solver.state.numberDensity.var)
    #         nTotal = np.sum(nsp[r], axis=1)
    #         for sp in range(solver.state.nSpecies):
    #             Xsp[r][:,sp] = nsp[r][:,sp] / nTotal
    #         if (r > 0):
    #             continuity[r] = (rho[r] - rho[r-1]) / dt + solver.grid.derivative.dot(rho[r] * vel[r][:,-1])
    #             continuity[r] += 2.0 * solver.grid.radGrad / solver.grid.radius * rho[r] * vel[r][:,-1]
                
    #         if (heatSwitch):
    #             heat[r] = heatSrc.compute(solver.state)[:,0] * solver.state.q0
    #         if (wallSwitch):
    #             wall[r] = wallSrc.compute(solver.state)[:,0] * solver.state.q0
    #         if (necSwitch):
    #             nec[r] = necSrc.compute(solver.state)[:,0] * solver.state.q0
    #         plasmaPower[r] = np.sum(solver.grid.norm.dot((heat[r] + wall[r] + nec[r]) * area))
    #         if (chemSwitch):
    #             rxn[r], rxnb[r] = chem.computeRates(solver.state)
    #             ndots[r] = np.matmul(chem.creationStoich.T, rxn[r] - rxnb[r])

    #     if "exit_p" in out_qoi:
    #         qoi_val_r[group]["exit_p"][c, :] = hist[rf][-1,0]
    #     if "exit_d" in out_qoi:
    #         qoi_val_r[group]["exit_d"][c, :] = [rho[rf][-1], hist[rf][-1,-1]*1e5]
    #     if "exit_v" in out_qoi:
    #         qoi_val_r[group]["exit_v"][c, :] = vel[rf][-1,-1]
    #     if "exit_T" in out_qoi:
    #         qoi_val_r[group]["exit_T"][c, :] = [Th[rf][-1], Te[rf][-1]]
    #     if "exit_X" in out_qoi:
    #         qoi_val_r[group]["exit_X"][c, :] = Xsp[rf][-1,2:7]
    #     if "heat_dep" in out_qoi:
    #         qoi_val_r[group]["heat_dep"][c, :] = plasmaPower[rf] * 1e-3

    #     # if hist[rf][-1,0] == 0.:
    #     #     breakpoint()
    #     ######
    #     # NOTE NOTE NOTE LEFT OFF HERE, CONTINUE WITH SAMPLER OBJECT DUMP
    #     ######


    #     # set up plots (exit quantities)
    #     # Pressure
    #     # Density
    #     # Axial Velocity
    #     # Heavy/Electron Temp
    #     # Species Mole Fraction (log)
    #     # Heat Deposition
    #     # Reaction Rates
    #     # Species Mole Fraction (linear)
    #     if make_plots:
    #         pIdx = 0
    #         # pIdx = 1 + solver.state.nVel + solver.state.nTemp

    #         nFig = 8

    #         def testF(x):
    #             fig, axs = plt.subplots(nFig, figsize=(12,2.4 * nFig))
    #             axs[0].plot(solver.grid.xg, hist[x][:,pIdx], '-b')
    #         #     axs[0].plot(solver.grid.xg, div[x], '-b')
    #             axs[0].set_title('Hydrodynamic pressure')
    #             axs[0].set_xlabel('$z$ ($m$)')
    #             axs[0].set_ylabel('$p$ ($Pa$)')

    #         #     axs[1].plot(solver.grid.xg, hist[x][:,1:3], '-')
    #         #     axs[1].set_title('Radial/Azimuthal velocity')
    #         #     axs[1].set_xlabel('$z$ ($m$)')
    #         #     axs[1].set_ylabel('$u$ ($m/s$)')
    #         #     axs[1].legend(['u_r', 'u_z'], loc='upper left')
    #             axs[1].semilogy(solver.grid.xg, rho[x], '-')
    #             axs[1].semilogy(solver.grid.xg, hist[x][:,-1]*1e5, '--')
    #         #     axs[1].semilogy(solver.grid.xg, np.sum(nsp[x], axis=1), '-r')
    #             axs[1].set_title('Density')
    #             axs[1].set_xlabel('$z$ ($m$)')
    #             axs[1].set_ylabel('$\rho$ ($kg/m^3$)')
    #         #     axs[1].legend(['u_r', 'u_z'], loc='upper left')
                
    #         #     axs[2].plot(solver.grid.xg, hist[x][:,3], '-')
    #             axs[2].plot(solver.grid.xg, vel[x][:,-1], '-')
    #             axs[2].set_title('Axial velocity')
    #             axs[2].set_xlabel('$z$ ($m$)')
    #             axs[2].set_ylabel('$u_z$ ($m/s$)')
                
    #             axs[3].plot(solver.grid.xg, Th[x], '-b')
    #             axs[3].plot(solver.grid.xg, Te[x], '--r')
    #         #     axs[3].plot(solver.grid.xg, hist[x][:,5], '--r')
    #         #     axs[3].plot(solver.grid.xg, solver.state.p0 * np.ones(solver.grid.Nx,), '-b')
    #         #     axs[3].set_ylim([0.299e3, 3.01e2])
    #         #     axs[3].set_ylim([0, 0.003])
    #             axs[3].set_title('Temperatures')
    #             axs[3].set_xlabel('$z$ ($m$)')
    #             axs[3].set_ylabel('$T$ ($K$)')
    #             axs[3].legend(['Th', 'Te'], loc='upper left')
                
    #             axs[4].semilogy(solver.grid.xg, (Xsp[x][:,1]), '-r')
    #             axs[4].semilogy(solver.grid.xg, (Xsp[x][:,2]), '-g')
    #             axs[4].semilogy(solver.grid.xg, (Xsp[x][:,3]), '-b')
    #             axs[4].semilogy(solver.grid.xg, (Xsp[x][:,4]), '-m')
    #             axs[4].semilogy(solver.grid.xg, (Xsp[x][:,5]), '-c')
    #             axs[4].semilogy(solver.grid.xg, (Xsp[x][:,6]), '-y')
    #             axs[4].set_title('Species mole fraction (log)')
    #             axs[4].set_xlabel('$z$ ($m$)')
    #             axs[4].set_ylabel('$X$')
    #             axs[4].legend(['e','Ar+','Arm','Arr','Arp','Arh'])
    #         #     axs[4].set_ylim([1e-11, 3e-7])
                
    #             axs[5].plot(solver.grid.xg, heat[x] * area * 1e-3, '-b')
    #             axs[5].plot(solver.grid.xg, wall[x] * area * 1e-3, '-r')
    #             axs[5].plot(solver.grid.xg, nec[x] * area * 1e-3, '-', color='orange')
    #         #     axs[5].set_ylim([0, 3.0e6])
    #             axs[5].set_title('heat deposition: %.3fkW' % (plasmaPower[x] * 1e-3))
    #             axs[5].set_xlabel('$z$ ($m$)')
    #             axs[5].set_ylabel('$q_e$ ($kW/m$)')
    #             axs[5].text(0.1, 0.8, 'Joule: %.3E kW' % (np.sum(solver.grid.norm.dot(heat[x] * area) * 1e-3)), color='b', transform=axs[5].transAxes)
    #             axs[5].text(0.1, 0.1, 'Radiation: %.3E kW' % (np.sum(solver.grid.norm.dot(nec[x] * area) * 1e-3)), color='orange', transform=axs[5].transAxes)
    #             axs[5].text(0.1, 0.3, 'Wall: %.3E kW' % (np.sum(solver.grid.norm.dot(wall[x] * area) * 1e-3)), color='r', transform=axs[5].transAxes)
                
    #             for r in range(numRxn):
    #                 rxnr = rxn[x].T[:,r]
    #                 rxnrb = rxnb[x].T[:,r]
    #         #         line = '--' if (r==2) else '-'
    #                 axs[6].semilogy(solver.grid.xg, rxnr, '-', label='%d'%r)
    #                 axs[6].semilogy(solver.grid.xg, rxnrb, '--', label='%d'%r)
    #         #     for sp in range(solver.state.nSpecies):
    #         #         ndotsp = ndots[x].T[:,sp]
    #         # #         line = '--' if (r==2) else '-'
    #         #         axs[6].plot(solver.grid.xg, ndotsp)
    #             axs[6].set_title('Reaction rates')
    #             axs[6].set_xlabel('$z$ ($m$)')
    #             axs[6].set_ylim([1e-15, 1e3])
    #             axs[6].set_ylabel('$R_f$ ($mol/m^3\cdot s$)')
    #         #     axs[6].legend(['Ar','e','Ar+','Ar*'])

    #             axs[7].plot(solver.grid.xg, (Xsp[x][:,1]), '-r')
    #             axs[7].plot(solver.grid.xg, (Xsp[x][:,2]), '-g')
    #             axs[7].plot(solver.grid.xg, (Xsp[x][:,3]), '-b')
    #             axs[7].plot(solver.grid.xg, (Xsp[x][:,4]), '-m')
    #             axs[7].plot(solver.grid.xg, (Xsp[x][:,5]), '-c')
    #             axs[7].plot(solver.grid.xg, (Xsp[x][:,6]), '-y')
    #         #     axs[7].plot(solver.grid.xg, (Xsp[x][:,3]), '-b')
    #         #     axs[7].set_ylim([0.0, 1e-6])
    #             axs[7].set_title('Species mole fraction')
    #             axs[7].set_xlabel('$z$ ($m$)')
    #         #     axs[7].set_ylabel('$n$ ($mol/m^3$)')
    #             axs[7].set_ylabel('$X$')
    #             axs[7].legend(['e','Ar+','Arm','Arr','Arp','Arh'])
                
    #             for ax in axs:
    #                 ax2 = ax.twinx()
    #                 ax2.plot(solver.grid.xg, solver.grid.radius, '-' ,linewidth=5.0, alpha=0.4)
    #                 ax2.set_ylabel('$R(z)$ ($m$)')
    #                 ax2.set_ylim([0, 0.03])

    #             fig.tight_layout()
                
    #             fig.savefig(r_dir + '/plots/post_results.png')
    #         #     plt.figure(2)
    #         #     plt.semilogy(solver.grid.xg, hist[x][:,-2:], '-')
    #         #     plt.plot(solver.grid.xg, rho[x] * vel[x][:,-1] * solver.grid.radius / visc[x], '-')
    #         #     plt.plot(solver.grid.xg, solver.state.CP * rho[x] * visc[x] / kh[x], '-')
    #         #     plt.ylim([1e-4, 1e2])
                
    #         #     plt.plot(solver.grid.xg, -solver.grid.derivative.dot(hist[x][:, pIdx]), '-b')
    #         #     plt.plot(solver.grid.xg, -solver.grid.derivative.dot(hist[x][:, solver.state.nVel]*vel[x][:,-1]), '-r')
    #         #     plt.plot(solver.grid.xg, -2.0 * solver.grid.radGrad / solver.grid.radius * hist[x][:, solver.state.nVel]*vel[x][:,-1], '--r')
    #         #     plt.plot(solver.grid.xg, -2.0 * solver.grid.radGrad / solver.grid.radius * hist[x][:, pIdx], '--b')
                
    #         #     plt.figure(3)
    #         #     for r in range(numRxn):
    #         #         rxnr = rxn[x].T[:,r]
    #         #         line = '--' if (r==2) else '-'
    #         #         plt.semilogy(solver.grid.xg, rxnr, line, label=rxnEqn[r])
    #         #     plt.legend()


    #         #     plt.plot(testGrid.xg, hist[x][:,1+testState.nVel+testState.nTemp], '-')
    #             return

    #         # NOTE: looking at last timestep for now
    #         testF(rf)
    #         # breakpoint()
    #         print("Maximum Electron Temperature")
    #         print(np.amax(Te[rf]))
    #         # testT = np.linspace(300, 11000, 500)
    #         # testT = Th[0]
    #         # ne = necSrc.nec.var(testT)
    #         # plt.figure(1)
    #         # plt.semilogy(testT, ne)
    #         # plt.semilogy(necSrc.nec.xTable,necSrc.nec.fTable,'.')

    #         # plt.figure(2)
    #         # plt.plot(solver.grid.xg, testT, '.-b',
    #         #         solver.grid.xg, ne / np.amax(ne) * np.amax(testT), '.-r')

    #         # plt.figure(3)
    #         # plt.plot(solver.grid.xg, ne, '.-')

            
        

    #     # filename = r_dir +  "/Torch1D.radius.txt"
    #     # data = np.array([solver.grid.xg, solver.grid.radius]).T
    #     # np.savetxt(filename, data)

    #     # filename = r_dir + "/Torch1D.temperature.txt"
    #     # data = np.array([solver.grid.xg, Th[-1], Te[-1]]).T
    #     # np.savetxt(filename, data)

    #     # filename = r_dir + "/Torch1D.mole_fraction.txt"
    #     # data = np.array([solver.grid.xg] + [Xsp[-1][:,k] for k in range(solver.state.nSpecies)]).T
    #     # np.savetxt(filename, data)

    #     # filename = r_dir + "/Torch1D.power.txt"
    #     # data = np.array([solver.grid.xg, heat[-1] * area, wall[-1] * area, nec[-1] * area]).T
    #     # np.savetxt(filename, data)
    #     # breakpoint()



    #     # Fluid
    #     if make_plots:
    #         # more plots
    #         # Reynolds Number
    #         # Prandtl Number
    #         # Heat Transfer Coef
    #         def testF(x):
    #             fig, axs = plt.subplots(nFig, figsize=(12,4. * nFig))

    #             axs[0].plot(solver.grid.xg, rho[x] * vel[x][:,-1] * solver.grid.radius / visc[x], '-')
    #             axs[0].set_xlabel('$z$ ($m$)')
    #             axs[0].set_ylabel('$Re_R$')

    #             axs[1].plot(solver.grid.xg, Pr[x], '-')
    #             axs[1].set_xlabel('$z$ ($m$)')
    #             axs[1].set_ylabel('$Pr$')

    #             ReL = solver.grid.L * vel[x][:,-1] * rho[x] / visc[x]
    #             NuL = 0.646 * np.power(Pr[x], (1./3.)) * np.sqrt(ReL)
    #             axs[2].plot(solver.grid.xg, heatTrnsf[x], '-', label='laminar fully-developed tube')
    #             axs[2].plot(solver.grid.xg, NuL * kh[x] / solver.grid.L, '-', label='laminar flat bndry, $Pr>0.5$')
    # #             axs[2].plot(solver.grid.xg, NuL, '-')
    # #             axs[2].plot(solver.grid.xg, 0.332 * np.power(Pr[x], (1./3.)) * np.sqrt(Rex), '-', label='laminar flat bndry, $Pr>0.5$')
    #             axs[2].set_title('Heat-transfer Coefficient')
    #             axs[2].set_xlabel('$z$ ($m$)')
    #             axs[2].set_ylabel('$h$')
    #             axs[2].set_xlim([0.0, np.amax(solver.grid.xg)])
    # #             axs[2].set_ylim([0.0, 20.0])
    #             axs[2].legend()

    #             for ax in axs:
    #                 ax2 = ax.twinx()
    #                 ax2.plot(solver.grid.xg, solver.grid.radius, '-' ,linewidth=5.0, alpha=0.4)
    #                 ax2.set_ylabel('$R(z)$ ($m$)')
    #                 ax2.set_ylim([0, 0.03])

    #             fig.tight_layout()
    #             fig.savefig(r_dir + '/plots/post_fluid_results.png')

    # #             plt.figure(2)
    # #             plt.semilogy(solver.grid.xg, hist[x]hist[-1][-1,0][:,-2:], '-')
    # #             plt.plot(solver.grid.xg, rho[x] * vel[x][:,-1] * solver.grid.radius / visc[x], '-')
    # #             plt.plot(solver.grid.xg, solver.state.CP * rho[x] * visc[x] / kh[x], '-')
    # #             plt.ylim([1e-4, 1e2])

    # #             plt.plot(solver.grid.xg, -solver.grid.derivative.dot(hist[x][:, pIdx]), '-b')
    # #             plt.plot(solver.grid.xg, -solver.grid.derivative.dot(hist[x][:, solver.state.nVel]*vel[x][:,-1]), '-r')
    # #             plt.plot(solver.grid.xg, -2.0 * solver.grid.radGrad / solver.grid.radius * hist[x][:, solver.state.nVel]*vel[x][:,-1], '--r')
    # #             plt.plot(solver.grid.xg, -2.0 * solver.grid.radGrad / solver.grid.radius * hist[x][:, pIdx], '--b')

    # #             plt.figure(3)
    # #             for r in range(numRxn):
    # #                 rxnr = rxn[x].T[:,r]
    # #                 line = '--' if (r==2) else '-'
    # #                 plt.semilogy(solver.grid.xg, rxnr, line, label=rxnEqn[r])
    # #             plt.legend()


    # #             plt.plot(testGrid.xg, hist[x][:,1+testState.nVel+testState.nTemp], '-')
    #             return

    #         testF(-1)

    #         print("Exit Electron Temperature")
    #         print(Te[-2][-1])
    #         print("Mole Fractions")
    #         print("Ion")
    #         print(Xsp[-2][-1,2])
    #         print("Meta")
    #         print(Xsp[-2][-1,3])
    #         print("Res")
    #         print(Xsp[-2][-1,4])
    #         print("Fourp")
    #         print(Xsp[-2][-1,5])
    #         print("Higher")
    #         print(Xsp[-2][-1,6])
    #         print("Minimum Prandtl Number")
    #         print(np.amin(Pr[-2]))


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