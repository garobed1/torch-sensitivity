import pickle
import os, sys
import numpy as np

sys.path.insert(0, '/g/g14/bedonian1/torch1d/')
import inputs
from axial_torch import AxialTorch


"""
Special script for a one-time transcription error


"""

home = os.getenv('HOME')
sample_out_dir = home + "/bedonian1/torch1d_resample_sens_r8/"
out_dir = home + "/bedonian1/torch1d_post_sens_r8/"
infile_name = "/torch1d_input.yml"

groups = ['AB','A', 'B']

fstep = 70000 

broken_cases = [['B', 699], ['B', 698], ['B', 700]] # if i recall correctly


out_qoi = ['exit_p', 'exit_d', 'exit_v', 'exit_T', 'exit_X', 'heat_dep']
qoi_sizes = {
    'exit_p': 1, 
    'exit_d': 2, 
    'exit_v': 1,
    'exit_T': 2,
    'exit_X': 5,
    'heat_dep': 1
}

#################################
# load outputs
qoi_val = {}
fqoin = out_dir + '/qoi_list.pickle' 
with open(fqoin, 'rb') as f:
    qoi_list = pickle.load(f)

for group in groups:

    fqoiv = out_dir + f'/qoi_samples_{group}.pickle' 

    with open(fqoiv, 'rb') as f:
        qoi_val[group] = pickle.load(f)


def listdir_nopickle(path):
    return [f for f in os.listdir(path) if not f.endswith('.pickle')]


# get sample directories
clist = listdir_nopickle(sample_out_dir)
clist.sort()

clists = {}
for group in groups:
    clists[group] = []

    while clist and clist[0].split('_')[1] == group:
        clists[group].append(clist.pop(0))

    # N[group] = len(clists[group])

for case in broken_cases:
    print(qoi_val[case[0]]['exit_p'][case[1]])
breakpoint()
# process broken cases
for case in broken_cases:

    group = case[0]
    c_ind = case[1]

    c_dir = sample_out_dir + '/' + clists[group][c_ind ]
    c_inf = c_dir + infile_name
    # r_dir = res_dir + '/' + clists[group][c_ind]


    args = inputs.parser.parse_args([c_inf])

    config = inputs.InputParser(args.input_file)
    config.dict_['state']['sanity_check'] = False

    solverType = config.getInput(['system','type'], fallback='axial-torch')
    solverDict = {'axial-torch': AxialTorch}

    # solverType = config.getInput(['system','type'], fallback='axial-torch')
    solver = solverDict[solverType](config)
    print(solver.state.__dict__.keys())
    nVel = solver.state.nVel

    nt0 = solver.state.timestep

    # breakpoint()

    # set up residual, and get "final" time step
    from os.path import exists
    rf = None
    Nr = divmod(solver.timeIntegrator.Nt, solver.outputFrequency)[0]
    filenames = []
    for r in range(Nr+1):
        filenames += ['%s/%s-%08d.h5' % (solver.outputDir, solver.prefix, nt0 + r * solver.outputFrequency)]
        
        if nt0 + r * solver.outputFrequency == fstep and exists(filenames[-1]): 
            rf = r + 0
    # breakpoint()
    crashFile = '%s/%s.crashed.h5' % (solver.outputDir, solver.prefix)
    if (exists(crashFile)):
        filenames += [crashFile]
        Nr += 1

    if rf is None and skip_incomplete:
        continue
    elif rf is None and not skip_incomplete:
        rf = -1

    hist = np.zeros([Nr+1, solver.grid.Nx, solver.state.nUnknowns])
    dt = solver.timeIntegrator.dt

    area = np.pi * (solver.grid.radius ** 2)
    heatSwitch = config.getInput(['system','heat-source','enabled'], fallback=False)
    wallSwitch = config.getInput(['system','wall-contribution','enabled'], fallback=False)
    necSwitch = config.getInput(['system','net-emission','enabled'], fallback=False)
    numRxn = len(config.getInput(['reactions'], fallback=[]))
    chemSwitch = (numRxn > 0)
    rxnEqn = []
    if (heatSwitch):
        from axial_torch_rhs2 import AxialTorchHeatSource
        heatSrc = AxialTorchHeatSource(solver.state, config)
    if (wallSwitch):
        from axial_torch_rhs2 import AxialTorchWallContribution
        wallSrc = AxialTorchWallContribution(solver.state, config)
    if (necSwitch):
        from axial_torch_rhs2 import AxialTorchNetEmission
        necSrc = AxialTorchNetEmission(solver.state, config)
    if (chemSwitch):
        from reaction import MassActionLaw
        chem = MassActionLaw(solver.state, config)
        rxnList = config.getInput(['reactions'], fallback=[])
        for rxn in rxnList:
            rxnEqn += [rxn['equation']]

    from axial_torch_rhs import AxialTorchDivergence
    divRhs = AxialTorchDivergence(solver.state)

    # get states
    div = np.zeros([Nr+1, solver.grid.Nx])
    rho = np.zeros([Nr+1, solver.grid.Nx])
    Th = np.zeros([Nr+1, solver.grid.Nx])
    Te = np.zeros([Nr+1, solver.grid.Nx])
    vel = np.zeros([Nr+1, solver.grid.Nx, solver.state.nVel])
    nsp = np.zeros([Nr+1, solver.grid.Nx, solver.state.nSpecies])
    Xsp = np.zeros([Nr+1, solver.grid.Nx, solver.state.nSpecies])
    continuity = np.zeros([Nr+1, solver.grid.Nx])
    plasmaPower = np.zeros(Nr+1,)
    heat = np.zeros([Nr+1, solver.grid.Nx])
    wall = np.zeros([Nr+1, solver.grid.Nx])
    nec = np.zeros([Nr+1, solver.grid.Nx])
    if (chemSwitch):
        rxn = np.zeros([Nr+1, chem.nRxn, solver.grid.Nx])
        rxnb = np.zeros([Nr+1, chem.nRxn, solver.grid.Nx])
        ndots = np.zeros([Nr+1, solver.state.nSpecies, solver.grid.Nx])

    if (len(config.getInput(['reactions'], fallback=[])) > 0):
        from reaction import MassActionLaw
        chem = MassActionLaw(solver.state, config)
    for r, filename in enumerate(filenames):
    #     filename = '%s/%s-%08d.h5' % (solver.outputDir, solver.prefix, nt0 + r * solver.outputFrequency)
        solver.state.loadState(filename)
        # solver.rhs.collInt.update(solver.state)
        solver.state.collInt.update(solver.state)
    #     transport.update(solver.state)
        hist[r] = np.copy(solver.state.conserved)
    #     solver.state.conserved = np.copy(hist[r])
    #     solver.state.time = r * solver.outputFrequency * dt
    # #     print(solver.state.time)
    #     solver.state.update()
    #     rhs0 = solver.rhs.compute(solver.state)
        div[r] = -np.copy(divRhs.compute(solver.state)[:,0])
    #     rho[r] = np.copy(solver.state.density.var)
        rho[r] = np.copy(solver.state.conserved[:,1+nVel])
        vel[r] = np.copy(solver.state.velocity.var)
        Th[r] = np.copy(solver.state.Th.var)
        Te[r] = np.copy(solver.state.Te.var)
        nsp[r] = np.copy(solver.state.numberDensity.var)
        nTotal = np.sum(nsp[r], axis=1)
        for sp in range(solver.state.nSpecies):
            Xsp[r][:,sp] = nsp[r][:,sp] / nTotal
        if (r > 0):
            continuity[r] = (rho[r] - rho[r-1]) / dt + solver.grid.derivative.dot(rho[r] * vel[r][:,-1])
            continuity[r] += 2.0 * solver.grid.radGrad / solver.grid.radius * rho[r] * vel[r][:,-1]
            
        if (heatSwitch):
            heat[r] = heatSrc.compute(solver.state)[:,0] * solver.state.q0
        if (wallSwitch):
            wall[r] = wallSrc.compute(solver.state)[:,0] * solver.state.q0
        if (necSwitch):
            nec[r] = necSrc.compute(solver.state)[:,0] * solver.state.q0
        plasmaPower[r] = np.sum(solver.grid.norm.dot((heat[r] + wall[r] + nec[r]) * area))
        if (chemSwitch):
            rxn[r], rxnb[r] = chem.computeRates(solver.state)
            ndots[r] = np.matmul(chem.creationStoich.T, rxn[r] - rxnb[r])

    if "exit_p" in out_qoi:
        qoi_val[group]["exit_p"][c_ind, :] = hist[rf][-1,0]
    if "exit_d" in out_qoi:
        qoi_val[group]["exit_d"][c_ind, :] = [rho[rf][-1], hist[rf][-1,-1]*1e5]
    if "exit_v" in out_qoi:
        qoi_val[group]["exit_v"][c_ind, :] = vel[rf][-1,-1]
    if "exit_T" in out_qoi:
        qoi_val[group]["exit_T"][c_ind, :] = [Th[rf][-1], Te[rf][-1]]
    if "exit_X" in out_qoi:
        qoi_val[group]["exit_X"][c_ind, :] = Xsp[rf][-1,2:7]
    if "heat_dep" in out_qoi:
        qoi_val[group]["heat_dep"][c_ind, :] = plasmaPower[rf] * 1e-3


for case in broken_cases:
    print(qoi_val[case[0]]['exit_p'][case[1]])

# this seems to fix it, but no idea why 698 took on 699's values

group = 'B'
filename = out_dir + f'/qoi_samples_{group}.pickle' 

with open(filename, 'wb') as f:

    pickle.dump(qoi_val[group], f)