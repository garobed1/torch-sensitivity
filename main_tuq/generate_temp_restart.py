import h5py as h5
import numpy as np

"""
Modify a tps restart file to include turbulent scalar quantities introduced by the K-e-zeta-f model

k, epsilon, zeta, f, and mut

Setting all to default initial or minimum values prescribed in zetaModel.cpp
"""


source_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/DEVCOLD_restart_output-torch-cold.sol.h5"
result_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/DEVRAMP_restart_output-torch-ramp.sol.h5"

add_swirl = False

s_attr = {}
with h5.File(source_restart, "r") as f:
    for key in f.attrs.keys():
        s_attr[key] = f.attrs[key]

    s_swirl = np.copy(f['swirl']['swirl'])
    # s_temperature = np.copy(f['temperature']['temperature'])
    s_velx = np.copy(f['velocity']['x-comp'])
    s_vely = np.copy(f['velocity']['y-comp'])
    s_tke = np.copy(f['tke']['tke'])
    s_tdr = np.copy(f['tdr']['tdr'])
    s_v2 = np.copy(f['v2']['v2'])
    s_zeta = np.copy(f['zeta']['zeta'])
    s_mut = np.copy(f['muT']['muT'])

temp = 300.0
# temp = 5500.0

temp_field = temp*np.ones_like(s_velx)

with h5.File(result_restart, "w") as f:
    for key in s_attr.keys():
        f.attrs[key] = s_attr[key]

    f.attrs['iteration'] = 0

    g1 = f.create_group("swirl")
    g1.create_dataset("swirl", s_swirl.shape, data=s_swirl)
    g2 = f.create_group("temperature")
    g2.create_dataset("temperature", temp_field.shape, data=temp_field)
    g3 = f.create_group("velocity")
    g3.create_dataset("x-comp", s_velx.shape, data=s_velx)
    g3.create_dataset("y-comp", s_vely.shape, data=s_vely)
    
    # turbulence model scalars
    g4 = f.create_group("tke")
    g4.create_dataset("tke", s_tke.shape, data=s_tke)
    g4 = f.create_group("tdr")
    g4.create_dataset("tdr", s_tke.shape, data=s_tdr)
    g4 = f.create_group("v2")
    g4.create_dataset("v2", s_tke.shape, data=s_v2)
    g4 = f.create_group("zeta")
    g4.create_dataset("zeta", s_tke.shape, data=s_zeta)
    g4 = f.create_group("muT")
    g4.create_dataset("muT", s_tke.shape, data=s_mut)