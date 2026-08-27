
import numpy as np
from scipy.io import loadmat
from scipy.special import erf, wofz
from scipy.interpolate import interpn
import os

# emulation of line-of-sight (LOS) effects on electron density measurements, applied to TPS results for direct comparison
# python adaptation of Dan Fries code in matlab, LOSevaluation.m
# idea is to provide functions to use as an option in sf_tps2d 

home = os.getenv("HOME")

kB = 8.617333262e-5 # [eV/K] 
eCharge = 1.60217663e-19 #% [Coloumbs]
eMass = 9.1093837139e-31 # kg 

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
Mi = 40 # particle mass for perturber ion [g/mol], should be argon or nitrogen
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
wvlNew = np.concatenate([wvlNew, -np.flip(wvlNew)[1:]])

# breakpoint()
"""
Ye_data : 1D array of radial field electron mass fraction data at an axial station
T_data : 1D array of radial field temperature data at an axial station
rho_data : 1D array of radial field density data at an axial station
"""
def getLOSeffect(Ye_data, T_data, rho_data, pos_data):

    
    lineShape = []

    # integrate over radial positions
    for rr in range(len(Ye_data)): # radial positions
        YeBroad = Ye_data[rr]
        TBroad = T_data[rr]
        rhoBroad = rho_data[rr]

        n_e = rhoBroad*YeBroad/eMass

        if lineCase == 'beta':
            
            dw_DopplerFit = dopplerBroad(lineCent,TBroad,Mi) # [nm]
            fitSpec = convLineShapeBeta(wvlNew,dw_inst,dw_DopplerFit,mu,TBroad,n_e)
        
            Qbalmer = 32*np.exp(-eupper/(kB*TBroad))
            

        # area normalize the line shape
        fitSpec = fitSpec/np.trapz(fitSpec, x=wvlNew)
        
        # compute EQL population of relevant H-atom levels
        Qelec = 2 + 8 * np.exp(-13.6*(1 - 1/4)/(kB*TBroad)) + 18 * np.exp(-13.6*(1 - 1/9)/(kB*TBroad))
        nj = rhoBroad*Qbalmer/Qelec * 0.02 # [1/m^3], 1% of H2, ~2% of H
        
        # compute line strength of transition (assume it is optically thin)
        emissCoeff = nj * A_ji * eij * (eCharge*1e-9) # [W/s.m^3.ster.nm]
        
        # scale line strength
        lineShape.append(fitSpec.T*emissCoeff)



    lineShape = np.array(lineShape)
    lineShape[np.isnan(lineShape)] = 0

    dr = [pos_data[n+1] - pos_data[n] for n in range(len(pos_data)-1)]
    dr.append(pos_data[-1] - pos_data[-2])
    losLine = 2*np.sum(lineShape, axis = 1)*dr
    losLine = losLine/max(losLine)
    losLine[np.isnan(losLine)] = 0
    
    neMaxIdx = np.argmax(n_e)
    neMaxProfile = lineShape[:,neMaxIdx]/max(lineShape[:,neMaxIdx])
    neMaxProfile[np.isnan(neMaxProfile)] = 0

    residual = neMaxProfile-losLine
    
    # Build output array
    Aout = [losLine, neMaxProfile]
    breakpoint()
    return Aout

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

        return convProfile
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
        starkProfile = interpn(points = (alphaVec[0],logNeVec[0],muVec[0],rhoVec[0]),
                                values = starkLib, 
                                xi = xigrid[0].T,
                                method='cubic',bounds_error = False, fill_value=0)
        starkProfile = np.concatenate([np.flipud(starkProfile[1:]).T, starkProfile.T])
        starkProfile = starkProfile/F0
        starkProfile = starkProfile/max(starkProfile)
    
        convolution = np.fft.fft(instProfile)*np.fft.fft(dopplerProfile)*np.fft.fft(starkProfile)
        convProfile = np.fft.ifft(convolution).real
        # breakpoint()
      
    return convProfile/max(convProfile)