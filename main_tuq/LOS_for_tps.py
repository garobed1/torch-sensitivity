
import numpy as np
from scipy.io import loadmat
from scipy.special import erf, wofz
from scipy.interpolate import interpn, RegularGridInterpolator    
from scipy.optimize import root
import os

# emulation of line-of-sight (LOS) effects on electron density measurements, applied to TPS results for direct comparison
# python adaptation of Dan Fries code in matlab, LOSevaluation.m
# idea is to provide functions to use as an option in sf_tps2d 

home = os.getenv("HOME")

kB = 8.617333262e-5 # [eV/K] 
eCharge = 1.60217663e-19 #% [Coloumbs]
eMass = 9.1093837139e-31 # kg 
arMass = 6.634e-26 # kg

# from Kepple and Griem "Improved Stark Profile Calculations for the
# Hydrogen Lines H α , H β , H γ , and H δ" 1968
TAlpha = np.array([5e3, 1e4, 2e4]) # [K]
nAlpha = np.array([1e15, 1e16, 1e17, 1e18])*1e6 # [1/m^3]
wParamAlpha = np.array([[0.00969, 0.0149, 0.0189, np.nan], [0.00777, 0.0134, 0.0186, 0.0215], [0.00601, 0.0114, 0.0175, 0.0226]]) # [Angs/cgs]

# argon is H_beta, focus on that
lineCase = 'beta' # fit Balmer alpha or beta line
gamma_inst = 0.0032 # instrument function Voigt/LT, Lorentz parameter
sigma_inst = 0.0082 # instrument function Voigt, Gauss parameter
dw_inst = [gamma_inst, sigma_inst] 
# dw_inst_est = 0.5346*2*gamma_inst + np.sqrt(0.2166*4*gamma_inst^2 + (2*sigma_inst*np.sqrt(2*np.log(2)))^2)
Mi = 40 # particle mass for perturber ion [g/mol], should be argon or nitrogen
# Ar is 40, N_2 is 28 (14 per atom)
Mh = 1 # particle mass for emitter [g/mol], should be atomic hydrogen
mu = 1/(1/Mi + 1/Mh)

if lineCase == "beta":
    betaLib = loadmat(home + "/starkBroadExp-main/GigosoStarkBroadeningProgram/ZikicGigososLib_2.mat")

    starkLib = np.array(betaLib['A2'])
    alphaVec = np.array(betaLib['alpha_1'])
    muVec = np.array(betaLib['mu_3'])
    logNeVec = np.array(betaLib['logNe_2'])
    rhoVec = np.array(betaLib['rho_4'])

    lineCent = 486.135
    eij = 12.7485 - 10.1988 # transition energy difference [eV]
    eupper = 12.7485 # upper state energy [eV]

    A_ji = 8.4193e+06/(4*np.pi)

wvlNew = np.linspace(-5,0,8000)
# wvlNew = np.linspace(-2,0,3000)
wvlNew = np.concatenate([wvlNew, -np.flip(wvlNew)[1:]])

data_interp = RegularGridInterpolator(points = (alphaVec[0],logNeVec[0],muVec[0],rhoVec[0]),
                                values = starkLib, 
                                method='cubic',bounds_error = False, fill_value=0)

# breakpoint()
"""
Ye_data : 1D array of radial field electron mass fraction data at an axial station
T_data : 1D array of radial field temperature data at an axial station
rho_data : 1D array of radial field density data at an axial station
"""
def getLOSeffect(Ye_data, T_data, rho_data, pos_data, ne_data=None, ntot_data=None):

    
    lineShape = []
    starkShape = []

    # integrate over radial positions
    for rr in range(len(pos_data)): # radial positions
        
        if ne_data is None:
            YeBroad = Ye_data[rr]
            TBroad = T_data[rr]
            rhoBroad = rho_data[rr]

            n_e = rhoBroad*YeBroad/eMass
        else:
            rhoBroad = ntot_data[rr]*arMass
            TBroad = T_data[rr]
            n_e = ne_data[rr]

        # breakpoint()
        if lineCase == 'beta':
            
            dw_DopplerFit = dopplerBroad(lineCent,TBroad,Mi) # [nm]
            fitSpec, starkSpec = convLineShapeBeta(wvlNew,dw_inst,dw_DopplerFit,mu,TBroad,n_e)
        
            Qbalmer = 32*np.exp(-eupper/(kB*TBroad))
            

        # area normalize the line shape
        fitSpec = fitSpec/np.trapz(fitSpec, x=wvlNew)
        
        # compute EQL population of relevant H-atom levels
        Qelec = 2 + 8 * np.exp(-13.6*(1 - 1/4)/(kB*TBroad)) + 18 * np.exp(-13.6*(1 - 1/9)/(kB*TBroad))

        if ne_data is None:
            nj = (rhoBroad/arMass)*Qbalmer/Qelec * 0.02 # [1/m^3], 1% of H2, ~2% of H
        else:
            nj = ntot_data[rr]*Qbalmer/Qelec * 0.02 # [1/m^3], 1% of H2, ~2% of H

        # NOTE: may be erroneous
        # nj = rhoBroad*Qbalmer/Qelec * 0.02 # [1/m^3], 1% of H2, ~2% of H
        
        # compute line strength of transition (assume it is optically thin)
        emissCoeff = nj * A_ji * eij * (eCharge*1e-9) # [W/s.m^3.ster.nm]
        
        # scale line strength
        lineShape.append(fitSpec.T*emissCoeff)
        # breakpoint()
        starkShape.append(starkSpec)

    lineShape = np.array(lineShape)
    starkShape = np.array(starkShape)
    lineShape[np.isnan(lineShape)] = 0
    starkShape[np.isnan(starkShape)] = 0

    dr = [pos_data[n+1] - pos_data[n] for n in range(len(pos_data)-1)]
    dr.append(pos_data[-1] - pos_data[-2])
    # losLine = 2*np.sum(lineShape*dr, axis = 0)
    # losLine = 2*np.einsum('ij,i->j', lineShape, dr)
    losLine = 2*np.einsum('ij,i->j', starkShape, dr)
    losLine = losLine/max(losLine)
    losLine[np.isnan(losLine)] = 0
    
    if ne_data is None:
        neMaxIdx = np.argmax(rho_data*Ye_data)
    else:
        neMaxIdx = np.argmax(ne_data)
    # neMaxProfile = lineShape[neMaxIdx,:]/max(lineShape[neMaxIdx,:])
    neMaxProfile = starkShape[neMaxIdx,:]/max(starkShape[neMaxIdx,:])
    # neMaxProfile = lineShape[neMaxIdx,:]
    neMaxProfile[np.isnan(neMaxProfile)] = 0

    residual = neMaxProfile-losLine
    
    # Build output array
    Aout = [wvlNew, losLine, neMaxProfile]

    # return Aout

    n_e_meas, n_e_max_meas = ProcessStark(Aout)

    # breakpoint()
    return n_e_meas, n_e_max_meas

## helper functions

def GaussFit(x,w): # Gaussian lineshape (instrument,Doppler)
    xbar = x*2/w
    gprofile = np.exp(-np.log(2)*(xbar**2))
    return gprofile/max(gprofile)

def LorentzFit(x,w): # Lorentzian lineshape (pressure,collision)
    xbar = x*2/w
    lprofile = 1./(1. + xbar**2)
    return lprofile/max(lprofile)

def Voigt(x,gamma,sigma): # Voigt lineshape (mix of Gaussian and Lorentzian)
    z = (x + 1j*gamma)/(np.sqrt(2) * sigma)
    profile = wofz(z).real/(np.sqrt(2) * sigma)
    return profile/max(profile)

def LTprofile(x,gamma,b,t): # Lorentzian-Trapezoid lineshape
    xbp = (x + b)/gamma
    xbm = (x - b)/gamma
    xtp = (x + t)/gamma
    xtm = (x - t)/gamma

    tbp = np.atan(xbp)
    tbm = np.atan(xbm)
    ttp = np.atan(xtp)
    ttm = np.atan(xtm)

    profile = 1./(np.pi * (b**2 - t**2)) * (x * (tbp + tbm - ttp - ttm) + 
        b * (tbp - tbm) - t * (ttp - ttm) + 
        gamma/2 * log((1 + xtm**2) * (1 + xtp**2) / (1 + xbm**2) / (1 + xbp**2)))
    return profile/max(profile)

def dopplerBroad(lambda0,Te,Me):
    # lambda0 - center wavelength of transition [nm]
    # Te - emitter temperature [K]
    # Me - emitter molar mass [g/mol]
    dw = 7.16e-7*lambda0*(Te/((Me/6.02214e23)/1.660599e-24))**0.5 # see NIST https://www.nist.gov/pml/atomic-spectroscopy-compendium-basic-ideas-notation-data-and-formulas/atomic-spectroscopy-6
    return dw

def convLineShapeBeta(wl,dw_inst,dw_doppler,mu,Te,ne):
    # instProfile = GaussFit(wl,dw_inst)
    instProfile = Voigt(wl,dw_inst[0],dw_inst[1])
    # instProfile = LTprofile(wl,dw_inst(1),dw_inst(2),dw_inst(3))
    dopplerProfile = GaussFit(wl,dw_doppler)
    
    if ne < 1e20:
        #convProfile = conv(instProfile,dopplerProfile,'same');
        convProfile = np.zeros(len(wl))

        return convProfile, convProfile
    else:
        # F0 = 2*pi*(4/15)^(2/3)*1.602e-19/(4*pi*8.854e-12)*(ne/1e6)^(2/3);
        F0 = 1.25e-9*(ne/1e6)**(2/3); # Holtsmark normal field, needs n_e in cm^-3 and is used in normalization of relative wavelength in Angstrom, Vidal et al. "HYDROGEN STARK-BROADENING TABLES" 1973
        zeroIdx = np.where(wl == 0)
        # breakpoint()
        alphaFit = wl[zeroIdx[0][0]:]*10/F0
        rho = ((ne)**(1/6))/(Te**0.5)/(((4*np.pi/3)**(1/3)) * (8.854e-12*1.381e-23/1.6021e-19**2)**0.5)
        xigrid = np.array([alphaFit,
                            np.log10(ne)*np.ones(alphaFit.shape[0]),
                            mu*np.ones(alphaFit.shape[0]),
                            rho*np.ones(alphaFit.shape[0])]),
        # starkProfile = interpn(points = (alphaVec[0],logNeVec[0],muVec[0],rhoVec[0]),
        #                         values = starkLib, 
        #                         xi = xigrid[0].T,
        #                         method='cubic',bounds_error = False, fill_value=0)
        starkProfile = data_interp(xigrid[0].T)
        starkProfile = np.concatenate([np.flipud(starkProfile[1:]).T, starkProfile.T])
        starkProfile = starkProfile/F0
        starkProfile = starkProfile/max(starkProfile)
    
        convolution = np.fft.fft(instProfile)*np.fft.fft(dopplerProfile)*np.fft.fft(starkProfile)
        convProfile = np.fft.ifft(convolution).real
        # breakpoint()
      
    return convProfile/max(convProfile), starkProfile


def areaFind(x,frac,wl,spec): # function to minimize to find full width at half area
    wlNew = np.linspace(wl[0],x,10000)
    specNew = np.interp(wlNew, wl, spec)
    res = np.trapz(specNew, x=wlNew, axis=0) - frac
    return res

def ProcessStark(A):

    # can skip many of the steps that the actual data process takes
    wvl = A[0]
    losLine = A[1]
    maxLine = A[2]

    # shift lines to taper at 0.
    losLine = losLine - min(losLine)
    losLine = losLine/max(losLine)
    maxLine = maxLine - min(maxLine)
    maxLine = maxLine/max(maxLine)

    # breakpoint()

    # compute FWHM
    losHalfIdx = np.argmin(abs(losLine - 0.5))
    losFWHM = 2*abs(wvl[losHalfIdx])
    maxHalfIdx = np.argmin(abs(maxLine - 0.5))
    maxFWHM = 2*abs(wvl[maxHalfIdx])

    # Compute electron number density from FWHA
    if lineCase == 'beta':
        # if strcmp(neCase,'approx')
        # starkProfileBeta = LorentzFit(wvl,losFWHM)
        # # starkProfileBeta = LorentzFit(wvl,dw_StarkFit(idxCnt))
        # totArea = np.trapz(starkProfileBeta, x=wvlNew); 
        # qArea = 0.25*totArea
        # areaFun_25 = lambda x : areaFind(x,qArea,wvlNew,starkProfileBeta)
        # wvl_25 = root(areaFun_25,-2.5).x
        # FWHA = 2*abs(wvl_25)
        # n_e = 1e23*(FWHA/1.666)**(1./0.68777) # Gigosos et al. "Computer simulated Balmer-alpha, -beta and -gamma Stark line profiles for non-equilibrium plasmas diagnostics" 2003
        # # rho_fit(idxCnt) = nan

        # starkProfileBetaMax = LorentzFit(wvl,maxFWHM)
        # # starkProfileBeta = LorentzFit(wvl,dw_StarkFit(idxCnt))
        # totAreaMax = np.trapz(starkProfileBetaMax, x=wvlNew); 
        # qAreaMax = 0.25*totAreaMax
        # areaFun_25_Max = lambda x : areaFind(x,qAreaMax,wvlNew,starkProfileBetaMax)
        # wvl_25_Max = root(areaFun_25_Max,-2.5).x
        # maxFWHA = 2*abs(wvl_25_Max)
        # n_e_max = 1e23*(maxFWHA/1.666)**(1./0.68777) # Gigosos et al. "Computer simulated Balmer-alpha, -beta and -gamma Stark line profiles for non-equilibrium plasmas diagnostics" 2003

        n_e = 1e23*(losFWHM/4.8)**(1./0.68116) # Gigosos et al. "Computer simulated Balmer-alpha, -beta and -gamma Stark line profiles for non-equilibrium plasmas diagnostics" 2003
        n_e_max = 1e23*(maxFWHM/4.8)**(1./0.68116) # Gigosos et al. "Computer simulated Balmer-alpha, -beta and -gamma Stark line profiles for non-equilibrium plasmas diagnostics" 2003


        # breakpoint()
    # breakpoint()


    return n_e, n_e_max
    # return n_e




# def resFunBeta(x,wl,data,dw_inst,lambda0,Me,mu,Te,wlNew,weightLim): # function to minimize for spectral fitting
#     # x[0] - Stark broadening FWHM
#     # x[1] - Ti for Doppler broadening
#     # x[2] - baseline shift
#     # data - data to fit to
#     # dw_inst - instrument function parameters
#     # lambda0 - center wavelength
#     # Mi - emitter molecular mass
#     # wl - data wavelength grid
#     # wlNew - finer wavelength grid

#     dw_Doppler = dopplerBroad(lambda0,x[1],Me) # Doppler broadening
#     muMod = mu*Te/x[1] # modified reduced mass due to ion dynamics (ion velocity proportionality), Zikic et al. "A program for the evaluation of electron number density from experimental hydrogen balmer beta line profiles" 2002

#     theory = convLineShapeBeta(wlNew,dw_inst,dw_Doppler,muMod,Te,x[0]) # compute theoretical lineshape
    
#     # compute residual
#     theoryInterp = np.interp(wlNew,theory,wl) + x(3)
#     theoryInterp = theoryInterp/max(theoryInterp)
#     res = data - theoryInterp
    
#     # add weights to residual
#     weights = ones(size(res))
#     maxDiff = 1 - min(data)
#     weightLimMod = weightLim*maxDiff + min(data)
#     weights(data < weightLimMod) = 0.5
#     res = res*sqrt(weights)

#     return res


if __name__ == "__main__":
    import csv
    import matplotlib.pyplot as plt

    TeFile = home + '/starkBroadExp-main/SimResults/2024-08-02/T.csv'
    neFile = home + '/starkBroadExp-main/SimResults/2024-08-02/n_e.csv'
    ntotFile = home + '/starkBroadExp-main/SimResults/2024-08-02/n_tot.csv'

    start = 3

    with open(TeFile, 'r') as f:
        Tdata = list(csv.reader(f))
        xT = np.array([float(x) for x in Tdata[0][1:]])
        rT = np.array([float(x) for x in [Tdata[y][0] for y in range(1,len(Tdata))]])
        dT = np.array(Tdata[1:][:], dtype=float)
        dT = dT[:,1:]

    with open(neFile, 'r') as f:
        nedata = list(csv.reader(f))

        xne = np.array([float(x) for x in nedata[0][1:]])
        rne = np.array([float(x) for x in [nedata[y][0] for y in range(1,len(nedata))]])
        dne = np.array(nedata[1:][:], dtype=float)
        dne = dne[:,1:]

    with open(ntotFile, 'r') as f:
        ntotdata = list(csv.reader(f))

        xntot = np.array([float(x) for x in ntotdata[0][1:]])
        rntot = np.array([float(x) for x in [ntotdata[y][0] for y in range(1,len(ntotdata))]])
        dntot = np.array(ntotdata[1:][:], dtype=float)
        dntot = dntot[:,1:]

    neLOS = np.zeros(xT.shape[0])
    neMAX = np.zeros(xT.shape[0])

    for i in range(start, len(xT)):
        print(i)
        neLOS[i], neMAX[i] = getLOSeffect(None, dT[:,i], None, rT, ne_data = dne[:,i], ntot_data = dntot[:,i])


    LTE_dat = np.vstack([xT, neLOS, neMAX]).T
    with open(home + "/torch-multifidelity/LTE_dat.npy", 'wb') as f:
        np.save(f, LTE_dat)


    # plt.plot(xT, neLOS, 'b')
    # plt.plot(xT, neMAX, 'r')
    # plt.savefig("showdatvalid_fix2.png", bbox_inches='tight')

    # breakpoint()
