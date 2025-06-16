import numpy as np
import h5py as h5
import pygmsh
import meshio
import matplotlib.pyplot as plt

"""
Script to generate a TPS mesh that smooths out the step at the core, intended
to test impact on solving TPS 7 species with 1st/2nd order finite elements

Test was unsuccessful, but we're leaving this script in the repo for reference

"""


### Mesh Files, Templates and Output
tps_mesh_f = '/g/g14/bedonian1/tps-inputs/axisymmetric/argon/mesh/torch-flow-refine.msh'
tps_mesh_c = '/g/g14/bedonian1/tps-inputs/axisymmetric/argon/mesh/torch-flow-refine-CC.msh'

# get profile of 2D mesh
tps_mesh = meshio.read(tps_mesh_f)
# tps_geom = pygmsh.built_in.Geometry()
rad = 0.002
# chamfer radius
rad2 = 0.002

def in_circ(point, loc):

    res = (point[0] - loc[0])**2 + (point[1] - loc[1])**2 - rad**2
    if res > 0:
        return False

    return True


def chamfer_shift(point, loc):

    center = np.array([loc[0] + rad2, loc[1] + rad2])

    angle = np.atan2(point[0] - center[0], point[1] - center[1])

    dest = np.array([center[0] + rad2*np.sin(angle), center[1] + rad2*np.cos(angle)])

    # proximity to wall, stagger interior point shift
    fac = np.sqrt(abs(min(min(point[0] - loc[0], 0), min(point[1] - loc[1], 0)))/rad)

    shift = (dest - point)*(1. - fac)
    print(fac)
    return shift


points = tps_mesh.points


# mask for points in the culprit corner
corner_loc = np.copy(points[7][:2])
corner_mask =  [i for i in range(points.shape[0]) if in_circ(points[i][:2], corner_loc)]

for i in corner_mask:
    points[i][:2] += chamfer_shift(points[i][:2], corner_loc)

# x_interp = x_0[extend_mask]
# r_f = r_0[:] 
# r_f[extend_mask] = np.interp(x_interp, torch_wall[:,1], torch_wall[:,0])
tps_mesh.points = points
tps_mesh.write(tps_mesh_c, file_format="gmsh22")