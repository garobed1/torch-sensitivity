import h5py as h5
import numpy as np



cold_restart = "/g/g14/bedonian1/bedonian1/tcal_cold_restart_output-torch.sol.h5"
hot_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_CONDP3_LTE/unrun_p3_restart_output-torch.sol.h5"

result_restart = "/g/g14/bedonian1/bedonian1/tcal_hot_restart_output-torch.sol.h5"
s_attr = {}
with h5.File(cold_restart, "r") as f:
    for key in f.attrs.keys():
        s_attr[key] = f.attrs[key]

    s_swirl = np.copy(f['swirl']['swirl'])
    s_temperature = np.copy(f['temperature']['temperature'])
    s_velx = np.copy(f['velocity']['x-comp'])
    s_vely = np.copy(f['velocity']['y-comp'])

with h5.File(hot_restart, "r") as f:

    h_temperature = np.copy(f['temperature']['temperature'])

s_attr['time'] = 0.0
s_attr['iteration'] = 0
s_temperature[:] = h_temperature[:]


with h5.File(result_restart, "w") as f:
    for key in s_attr.keys():
        f.attrs[key] = s_attr[key]

    g1 = f.create_group("swirl")
    g1.create_dataset("swirl", s_swirl.shape, data=s_swirl)
    g2 = f.create_group("temperature")
    g2.create_dataset("temperature", s_temperature.shape, data=s_temperature)
    g3 = f.create_group("velocity")
    g3.create_dataset("x-comp", s_velx.shape, data=s_velx)
    g3.create_dataset("y-comp", s_vely.shape, data=s_vely)