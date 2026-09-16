import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

home = os.getenv("HOME")

P = 3

tke_cold = home + "/bedonian1/mean_tps2d_newmesh/tps2d_zetaf_current_hotrun/inputs/tke_3d.csv"
tke_cold_smooth = home + "/bedonian1/mean_tps2d_newmesh/tps2d_zetaf_current_hotrun/inputs/tke_3d_smooth.csv"

# ind_bad = [1438, 1477]
# ind_bad = [1438, 1477]
# ind_bad = [1425, 1492]
# ind_bad = [1395, 1492]
ind_bad = [1435, 1492]
Lwind = 10
Rwind = 1

# open inputs
tcl = []
with open(tke_cold, newline='') as csvfile:
    fread = csv.reader(csvfile)
    for row in fread:
        tcl.append([float(x) for x in row])


tc = np.array(tcl)

xt = tc[:,0]
tt = tc[:,3]
vt = tc[:,4]

plt.plot(xt, tt)
plt.plot(xt, vt)

plt.savefig("tke_sharp.png")

xflo = xt[ind_bad[0]-Lwind:ind_bad[0]]
xfhi = xt[ind_bad[1]:ind_bad[1]+Rwind]
tflo = tt[ind_bad[0]-Lwind:ind_bad[0]]
tfhi = tt[ind_bad[1]:ind_bad[1]+Rwind]
vflo = vt[ind_bad[0]-Lwind:ind_bad[0]]
vfhi = vt[ind_bad[1]:ind_bad[1]+Rwind]
xint = xt[ind_bad[0]:ind_bad[1]]

xf = np.append(xflo, xfhi)
tf = np.append(tflo, tfhi)
vf = np.append(vflo, vfhi)

# tpval = np.polyfit(xf, tf, P)
# vpval = np.polyfit(xf, vf, P)

# tnew = np.polyval(tpval, xint)
# vnew = np.polyval(vpval, xint)

tpval = make_interp_spline(xf, tf, P)
vpval = make_interp_spline(xf, vf, P)

tnew = tpval(xint)
vnew = vpval(xint)


tt[ind_bad[0]:ind_bad[1]] = tnew
vt[ind_bad[0]:ind_bad[1]] = vnew

plt.plot(xt, tt)
plt.plot(xt, vt)

plt.savefig("tke_smooth.png")


with open(tke_cold_smooth, 'w', newline='') as csvfile:
    fwrite = csv.writer(csvfile)
    for i, row in enumerate(tcl):
        fwrite.writerow([row[0], row[1], row[2], tt[i], vt[i]])
breakpoint()