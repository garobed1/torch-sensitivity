import h5py as h5
import numpy as np


source_mu_kap_sig = "/g/g14/bedonian1/tps/test/lte-data/argon_transport_1atm.h5"
source_r_cp = "/g/g14/bedonian1/tps/test/lte-data/argon_thermo_1atm.h5"
result_file = "argon_lomach_1atm.h5"



with h5.File(source_mu_kap_sig, "r") as f:
    data_1 = f["T_mu_kappa_sigma"][:,:]

with h5.File(source_r_cp, "r") as f:
    data_2 = f["T_energy_R_c"][:,:]



data_res = np.zeros([data_1.shape[0], 6])

data_res[:,0] = data_1[:,0]
data_res[:,1] = data_1[:,1]
data_res[:,2] = data_1[:,2]
data_res[:,3] = data_1[:,3]

data_res[:,4] = data_2[:,2]
data_res[:,5] = data_2[:,3]

with h5.File(result_file, "w") as f:

    f.create_dataset("T_mu_kap_sig_R_Cp", data_res.shape, data=data_res)