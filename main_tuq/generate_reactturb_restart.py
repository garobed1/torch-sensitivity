import h5py as h5
import numpy as np

"""
Modify a tps restart file to include minimal species fractions

And copy over zeta f turbulent quantities
"""

# source_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_hot_down1cm_zetaf/DEVWQTKAPPA_restart_output-torch-hot-zeta-f-r2.sol.h5"
#result_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_hot_down1cm_zetaf/DEVWQTKAPPA_6species_restart_output-torch-hot-zeta-f-r2.sol.h5"
# result_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_hot_down1cm_zetaf/DEVWQTKAPPA_6speciesM_restart_output-torch-hot-zeta-f-r2.sol.h5"
# result_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_hot_down1cm_zetaf/DEVWQTKAPPA_6species3_restart_output-torch-hot-zeta-f-r2.sol.h5"
# result_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_hot_down1cm_zetaf/DEVWQTKAPPA_7species_restart_output-torch-hot-zeta-f-r2.sol.h5"
source_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/DEVRAMP8_restart_output-torch-kappa.sol.h5"
result_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_newmesh/mean_tps2d_v2_hot_down1cm_zetaf/DEVRAMP8_restart_output-torch-7sp.sol.h5"

seven_species = True
frac_init = 1e-10
frac_ion_init = 1e-6
#frac_init = 0.0
#frac_ion_init = 0.0
#frac_init = 1e-1
#frac_ion_init = 1e-1

frac_elec_init = frac_ion_init*(5.48579908782496e-7/39.948e-3)

s_attr = {}
with h5.File(source_restart, "r") as f:
    for key in f.attrs.keys():
        s_attr[key] = f.attrs[key]

    s_swirl = np.copy(f['swirl']['swirl'])
    s_temperature = np.copy(f['temperature']['temperature'])
    s_velx = np.copy(f['velocity']['x-comp'])
    s_vely = np.copy(f['velocity']['y-comp'])
    s_tke = np.copy(f['tke']['tke'])
    s_tdr = np.copy(f['tdr']['tdr'])
    s_v2 = np.copy(f['v2']['v2'])
    s_zeta = np.copy(f['zeta']['zeta'])
    s_mut = np.copy(f['muT']['muT'])

# now add species, see if uniform works
ar_m = frac_init*np.ones_like(s_swirl)
ar_r = frac_init*np.ones_like(s_swirl)
ar_p = frac_init*np.ones_like(s_swirl)
if seven_species:
    ar_h = frac_init*np.ones_like(s_swirl)
ar_i = frac_ion_init*np.ones_like(s_swirl)
elec = frac_elec_init*np.ones_like(s_swirl)

fac = 1.0
if seven_species:
    fac = 0.0
ar_g = (1.0 - (4. - fac)*frac_init - frac_ion_init - frac_elec_init)*np.ones_like(s_swirl)

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

    # species
    g5 = f.create_group("species")
    g5.create_dataset("Y_0", ar_m.shape, data=ar_m)
    g5.create_dataset("Y_1", ar_r.shape, data=ar_r)
    g5.create_dataset("Y_2", ar_p.shape, data=ar_p)

    if seven_species:
        g5.create_dataset("Y_3", ar_h.shape, data=ar_h)
        g5.create_dataset("Y_4", ar_i.shape, data=ar_i)
        g5.create_dataset("Y_5", elec.shape, data=elec)
        g5.create_dataset("Y_6", ar_g.shape, data=ar_g)
    else:
        g5.create_dataset("Y_3", ar_i.shape, data=ar_i)
        g5.create_dataset("Y_4", elec.shape, data=elec)
        g5.create_dataset("Y_5", ar_g.shape, data=ar_g)
