import h5py as h5
import numpy as np

"""
Modify a tps restart file to include turbulent scalar quantities introduced by the K-e-zeta-f model

k, epsilon, zeta, f, and mut

Setting all to default initial or minimum values prescribed in zetaModel.cpp
"""

# source_restart = "/g/g14/bedonian1/tps/test/restart_output-pipe-zeta-f-fdp-y1.sol.h5"
# result_restart = "/g/g14/bedonian1/tps/test/BASE_restart_output-pipe-zeta-f-fdp-fix-y1.sol.h5"
# source_restart = "/g/g14/bedonian1/tps/test/STEADY2_restart_output-pipe-zeta-f-fdp-fix-y1.sol.h5"
# result_restart = "/g/g14/bedonian1/tps/test/BASE_restart_output-pipe-zeta-f-fdp-swirl.sol.h5"
# source_restart = "/g/g14/bedonian1/tps/test/restart_output-pipe-lam-rm.sol.h5"
# result_restart = "/g/g14/bedonian1/tps/test/BASE_restart_output-pipe-zeta-f-swirl-rm.sol.h5"
# source_restart = "/g/g14/bedonian1/tps/test/BASE_restart_output-pipe-zeta-f-swirl-rm.sol.h5"
# result_restart = "/g/g14/bedonian1/tps/test/HBASE_restart_output-pipe-zeta-f-swirl-rm.sol.h5"
# source_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_cold_down1cm_zetaf/DEV2_restart_output-torch-zeta-f.sol.h5"
# result_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_cold_down1cm_zetaf/DEV2zf_restart_output-torch-zeta-f.sol.h5"
# source_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_hot_down1cm_zetaf/DEVBETA2_restart_output-torch-hot-beta2-zeta-f.sol.h5"
# result_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_hot_down1cm_zetaf/DEVBETA2zf_restart_output-torch-hot-beta2-zeta-f.sol.h5"
# source_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/DEVRAMP2_restart_output-torch-ramp.sol.h5"
# result_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/DEVRAMP3_restart_output-torch-ramp.sol.h5"
# source_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_s3_hot_down1cm_zetaf_rm10/DEV1_restart_output-torch-hot-rm10-2-alpha-zeta-f.sol.h5"
# result_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_s3_hot_down1cm_zetaf_rm10/DEV2_restart_output-torch-hot-rm10-2-alpha-zeta-f.sol.h5"
# source_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_s3_hot_down1cm_zetaf_rm11/DEV1_restart_output-torch-hot-rm11-2-alpha-zeta-f.sol.h5"
# result_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_s3_hot_down1cm_zetaf_rm11/DEV2_restart_output-torch-hot-rm11-2-alpha-zeta-f.sol.h5"

source_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/BADTURB6_restart_output-torch-cold-v2-rm11-2-fine.sol.h5"
result_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/restart_output-torch-cold-v2-rm11-2-fine.sol.h5"

add_swirl = False

s_attr = {}
with h5.File(source_restart, "r") as f:
    for key in f.attrs.keys():
        s_attr[key] = f.attrs[key]

    s_swirl = np.copy(f['swirl']['swirl'])
    s_temperature = np.copy(f['temperature']['temperature'])
    s_velx = np.copy(f['velocity']['x-comp'])
    s_vely = np.copy(f['velocity']['y-comp'])

# using initial values
rho = 1.629
# rho = 1.0
Cmu = 0.22
# tke_init = 0.1
# tdr_init = 0.01
tke_init = 1e-7
tdr_init = 1e-8
# tke_init = 1e-3
# tdr_init = 1e-4
two_thirds = 2./3.
f_min = 1e-12
# argon_visc = 2.1e-5
argon_visc = 1.0e-4
# S is unknown, set to 1

tke_field = tke_init*np.ones_like(s_velx)
tdr_field = tdr_init*np.ones_like(s_velx)
zeta_field = two_thirds*np.ones_like(s_velx)
zero_x = np.zeros_like(s_velx)
v2_field = tke_init*zeta_field
# f_field = f_min*np.ones_like(s_velx) # not quite minimum value

Tval = max(min(tke_init/tdr_init, 0.6/(np.sqrt(6)*Cmu*two_thirds)), 6*np.sqrt(argon_visc/tdr_init))

mut_field = rho*Cmu*zeta_field*tke_field*Tval

if add_swirl:
    n_x = 17
    # inferred x locations from exact pipe sol
    x_loc_inf = np.sqrt((-s_vely[:n_x] + 2.)/2. )
    n_y = s_vely.size//17
    p_swirl = -x_loc_inf*x_loc_inf + 1

    rt = 0.9
    R = 1.0
    u_th_max = 1.0

    for i in range(len(x_loc_inf)):
        x = x_loc_inf[i]
        if (x < rt):
            p_swirl[i] = u_th_max * x / rt
        else:
            p_swirl[i] = u_th_max * rt * (R - x) / (x * (R - rt))

    # for i in range(n_y):
    for i in range(1):
        s_swirl[i*n_x:(i+1)*n_x] += p_swirl
    # breakpoint()

with h5.File(result_restart, "w") as f:
    for key in s_attr.keys():
        f.attrs[key] = s_attr[key]

    g1 = f.create_group("swirl")
    g1.create_dataset("swirl", s_swirl.shape, data=s_swirl)
    g2 = f.create_group("temperature")
    g2.create_dataset("temperature", s_temperature.shape, data=s_temperature)
    g3 = f.create_group("velocity")
    g3.create_dataset("x-comp", s_velx.shape, data=s_velx)
    # g3.create_dataset("x-comp", zero_x.shape, data=zero_x)
    g3.create_dataset("y-comp", s_vely.shape, data=s_vely)
    
    # turbulence model scalars
    g4 = f.create_group("tke")
    g4.create_dataset("tke", tke_field.shape, data=tke_field)
    g4 = f.create_group("tdr")
    g4.create_dataset("tdr", tdr_field.shape, data=tdr_field)
    g4 = f.create_group("v2")
    g4.create_dataset("v2", v2_field.shape, data=v2_field)
    g4 = f.create_group("zeta")
    g4.create_dataset("zeta", zeta_field.shape, data=zeta_field)
    g4 = f.create_group("muT")
    g4.create_dataset("muT", mut_field.shape, data=mut_field)