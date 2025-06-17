import numpy as np
import h5py as h5
import pygmsh
import meshio
import matplotlib.pyplot as plt


"""
Script to generate a torch1d grid that better conforms to the axisymmetric
2D TPS mesh


"""


### Mesh Files, Templates and Output
t1d_mesh_f = '../trevilo-cases/torch_tst_2023/Torch.mesh.h5'
tps_mesh_f = '/g/g14/bedonian1/tps-inputs/axisymmetric/argon/mesh/torch-flow-refine.msh'
t1d_mesh_o = '../trevilo-cases/torch_tst_2023/Torch_V2.mesh.h5'


##########################################################################################################
# Script Starts Here
##########################################################################################################

with h5.File(t1d_mesh_f, 'r') as f:
    x_0 = f["x"][:]
    r_0 = f["radius"][:]

# get profile of 2D mesh
tps_mesh = meshio.read(tps_mesh_f)
# tps_geom = pygmsh.built_in.Geometry()

wall_zone = 8
torch_wall_mask = [x for x in range(len(tps_mesh.cell_data['gmsh:physical'][1])) if tps_mesh.cell_data['gmsh:physical'][1][x] == wall_zone]
wall_locs = tps_mesh.cells[1].data[torch_wall_mask]

torch_wall = tps_mesh.points[wall_locs[:,0]]
sarg = np.argsort(torch_wall[:,1])
torch_wall = torch_wall[sarg]

xdiff = x_0[-1] - torch_wall[-1,1]
torch_wall[:,1] += xdiff

plt.plot(x_0, r_0)
plt.plot(torch_wall[:,1], torch_wall[:,0])
plt.savefig("t1d_tps_before.png")
breakpoint()

# generate t1d radius from tps mesh on existing x points
# mask for x points shared between t1d and tps
extend_mask =  [x for x in range(len(x_0)) if x_0[x] > xdiff]
x_interp = x_0[extend_mask]
r_f = r_0[:] 
r_f[extend_mask] = np.interp(x_interp, torch_wall[:,1], torch_wall[:,0])


plt.plot(x_0, r_0)
plt.plot(torch_wall[:,1], torch_wall[:,0])
plt.savefig("t1d_tps_after.png")
breakpoint()
plt.plot(x_0, r_f)
# exc_sum = np.sum(table[:,7:], axis=1)
# table = table[:,:8]
# table[:,-1] = exc_sum

# fname = "AxialICPTorch_4s.ic.h5"

with h5.File(t1d_mesh_o, 'w') as f:
    dset = f.create_dataset("x", x_0.shape, data=x_0)
    dset = f.create_dataset("radius", r_f.shape, data=r_f)
