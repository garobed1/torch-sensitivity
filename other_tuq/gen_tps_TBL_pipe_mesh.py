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
tps_mesh_f = '/g/g14/bedonian1/tps/test/meshes/axi-pipe.msh'
tps_mesh_c = '/g/g14/bedonian1/tps/test/meshes/axi-pipe-wall-y1.msh'

# get profile of 2D mesh
tps_mesh = meshio.read(tps_mesh_f)
# tps_geom = pygmsh.built_in.Geometry()
decay = 1.0
wall_locs = [0.0, 1.0]
offset = 3e-3 # to achieve y+ \approx 1
# are we on either wall to the left or right?
def off_wall(point, locs):

    # left
    res1 = point[0] - locs[0]
    if res1 < 1e-10:
        return False
    
    # right
    res1 = point[0] - locs[1]
    if res1 > -1e-10:
        return False

    return True

def wall_shift(point, locs):
    # locs[0]: where to start shift
    # locs[1]: wall location
    diff = locs[1]-locs[0]
    sfac = 1./diff
    
    dest = -(sfac*(point[0] - locs[1]))**2 + 1 + offset
    # dest = -(sfac*pow(point[0] - locs[1], 2.5)) + 1
    dest = locs[0] + dest*diff

    shift = dest - point[0]
    print(f"{point[0]}, {dest}")
    return shift


points = tps_mesh.points


# mask for points between the walls
wall_mask =  [i for i in range(points.shape[0]) if off_wall(points[i], wall_locs)]

for i in wall_mask:
    points[i][0] += wall_shift(points[i], wall_locs)

# x_interp = x_0[extend_mask]
# r_f = r_0[:] 
# r_f[extend_mask] = np.interp(x_interp, torch_wall[:,1], torch_wall[:,0])
tps_mesh.points = points
tps_mesh.write(tps_mesh_c, file_format="gmsh22")