import numpy as np
import csv
import os
import matplotlib.pyplot as plt

home = os.getenv("HOME")


inlet_cold = home + "/bedonian1/mean_tps2d_newmesh/tps2d_zetaf_current_hotrun/inputs/inlet_axi2d_interp_down1cm_cold.csv"
inlet_hot = home + "/bedonian1/mean_tps2d_newmesh/tps2d_zetaf_current_hotrun/inputs/inlet_axi2d_interp_down1cm_hot.csv"

tke_cold = home + "/bedonian1/mean_tps2d_newmesh/tps2d_zetaf_current_hotrun/inputs/tke_3d.csv"
tke_hot = home + "/bedonian1/mean_tps2d_newmesh/tps2d_zetaf_current_hotrun/inputs/tke_3d_hot.csv"

# open inputs
ic = []
ih = []
tc = []
with open(inlet_cold, newline='') as csvfile:
    fread = csv.reader(csvfile)
    for row in fread:
        ic.append([float(x) for x in row])
with open(inlet_hot, newline='') as csvfile:
    fread = csv.reader(csvfile)
    for row in fread:
        ih.append([float(x) for x in row])
with open(tke_cold, newline='') as csvfile:
    fread = csv.reader(csvfile)
    for row in fread:
        tc.append([float(x) for x in row])

ic = np.array(ic)
ih = np.array(ih)
tc = np.array(tc)

xinlet_cold = ic[:,1]
vinlet_cold = ic[:,6]
xinlet_hot = ih[:,1]
vinlet_hot = ih[:,6]

xinlet_cold_tke = tc[:,0]
TKEinlet_cold_tke = tc[:,3]
V2inlet_cold_tke = tc[:,4]

vinlet_cold_interp_tke = np.interp(xinlet_cold_tke, xinlet_cold, vinlet_cold)
vinlet_hot_interp_tke = np.interp(xinlet_cold_tke, xinlet_hot, vinlet_hot)

fac_tke = abs(vinlet_cold_interp_tke[:-1])/TKEinlet_cold_tke[:-1]
fac_v2 = abs(vinlet_cold_interp_tke[:-1])/V2inlet_cold_tke[:-1]

plt.plot(xinlet_cold_tke[:-1], vinlet_cold_interp_tke[:-1])
plt.plot(xinlet_cold_tke[:-1], TKEinlet_cold_tke[:-1])
plt.plot(xinlet_cold_tke[:-1], V2inlet_cold_tke[:-1])
plt.plot(xinlet_cold_tke[:-1], vinlet_hot_interp_tke[:-1])
# plt.plot(xinlet_hot[:-1], vinlet_hot[:-1])
# plt.plot(xinlet_cold_tke[:-1], fac_tke)
# plt.plot(xinlet_cold_tke[:-1], fac_v2)
plt.savefig("fac.png")



breakpoint()