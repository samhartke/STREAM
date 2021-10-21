# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:17:25 2020

@author: Sam

This script uses CSGD error models trained over study area
to transform spatiotemporally correlation uniform noise into rainfall values
"""

from netCDF4 import Dataset
from datetime import date, timedelta
import numpy as np
import scipy as sp
from tqdm import tqdm
import time



#==============================================================================
# Function to retrieve wetted area ratio (WAR) across entire field
# - more efficient than retrieving pixel by pixel

area = np.zeros((100,150))
d = 21
area.fill((d)**2)

a = np.reshape(np.arange(11,d),(1,10))
b = np.reshape(np.arange(20,10,-1),(1,10))

area[:10,:10] = np.transpose(a)*a
area[90:,:10] = np.transpose(b)*a
area[:10,140:] = np.transpose(a)*b
area[90:,140:] = np.transpose(b)*b

for i in range(11,d):
    area[i-11,10:140] = i*d
    area[110-i,10:140] = i*d
    area[10:90,i-11] = i*d
    area[10:90,160-i] = i*d
    

def getWARfield(field,r):
    
    ysize = np.shape(field)[0]
    xsize = np.shape(field)[1]
    
    rainy = np.zeros(np.shape(field))
    rainy[field>=0.1] = 1 # delineate rainy pixels with value of 1
    
    rainysum = np.zeros(np.shape(field))
    

    for i in range(-r,r+1):
        
        irange = (np.max((0,-i)),np.min((xsize,xsize-i)))
        
        for j in range(-r,r+1):
            
            subfield = rainy[np.max((0,j)):np.min((ysize,ysize+j)),np.max((0,i)):np.min((xsize,xsize+i))]
            
            jrange = (np.max((0,-j)),np.min((ysize,ysize-j)))
            
            rainysum[jrange[0]:jrange[1],irange[0]:irange[1]] += subfield
    
    
    WARfield = rainysum/area
    
    return(WARfield)



# ----------------------------------------------------------------------------

def simulatePrecip(dt,n_ens,ts,obsFile,noiseFile,paramsFile):
    
    end_dt = dt + timedelta(hours=(ts-1)) # end date of simulation
    #print("Generating %d-member precip ensemble for %s - %s"%(n_ens,dt.strftime("%Y-%m-%d"),end_dt.strftime("%Y-%m-%d")))
    
    # ---  indicate whether to use correlated noise or not  ---
    corr = "imerg"   # corr can equal "imerg" or "none"
    
    
    # ---------------------  READ IN SATELLITE PRECIPITATION  -----------------
    i1 = (dt - date(dt.year,1,1)).days*24  # starting index of IMERG data for simulation period
    i2 = i1 + ts + 1  # ending index
    
    ds = Dataset(obsFile)
    
    obs = ds.variables['prcp'][:,:,i1:i2].astype('float32') # grab IMERG data from simulation period
    obs[obs<0.1] = 0.
    
    ysize = np.shape(obs)[0]
    xsize = np.shape(obs)[1]
    
    
    # ------ open up wetted area ratio covariate data for simulation period ----
    # ds = Dataset(wd + 'data/WetAR%d.r10.hourly.nc'%dt.year)
    # war = ds.variables['war'][:,:,i1:i2].astype('float32')
    
    war = np.zeros(np.shape(obs))
    
    for i in range(ts):
        
        war[:,:,i] = getWARfield(obs[:,:,i],10)
    
    
    # -------------------------  RETRIEVE NOISE  ------------------------------
    
    if corr=="imerg":
        
        ds = Dataset(noiseFile)
        q = ds.variables['q'][:n_ens,:,:,:].astype('float16')
        # Note: if no correlated noise exists for this time range, create some and 
        # save to netcdf by calling generateNoiseEnsemble.py
    
    
    elif corr=="none":
        
        # -----  generate uncorrelated (white) noise  -----
        q = np.random.uniform(size=(n_ens,ts,ysize,xsize)).astype('float16')
    
    
    
    # set upper threshold for uniform noise so that unrealistically extreme
    # values don't get selected from conditional distribution tails
    q[q>0.995] = 0.995
    
    
    
    
    # ---------- Read in parameter grids for precip error model ---------------       
    
    ds = Dataset(paramsFile)
    
    lin = False # indicate if error model uses linear or nonlinear regression
    
    climparams = np.empty((3,ysize,xsize))
    n=0
    for name in ('clim1','clim2','clim3'):
        climparams[n,:,:] = ds.variables[name][:,:]
        n+=1
    
    reg = np.empty((6,ysize,xsize))
    n=0
    for name in ('par1','par2','par3','par4','par5'):
        reg[n,:,:] = ds.variables[name][:,:]
        n+=1
    
    reg[5,:,:] = 0.
    
    imean = ds.variables['mean'][:,:]
    warmean = ds.variables['WARmean'][:,:]
    
    ds = None
    
    
    
    # -------------------- SIMULATE PRECIPITATION --------------------------------
    
    
    simPrcp = np.zeros(np.shape(q),dtype=np.float32) # create empty array to hold simulated precip
    
    
    # -- loop through each pixel in study area --
    for y in tqdm(range(ysize),desc="Generating %d-member precip ensemble"%n_ens):
                
        for x in range(xsize):
            obs_ = obs[y,x,:-1]
            war_ = war[y,x,:-1]
            
            qVals = q[:,:,y,x]
            
            mu_clim=climparams[:,y,x][0]
            sigma_clim=climparams[:,y,x][1]
            delta_clim=climparams[:,y,x][2]

            # note: this is currently using the equivalent of pcsgd(0.0,condparams0) when IMERG and WAR is zero, rather than pcsgd(0.1,condparams0)
            logarg=reg[:,y,x][1] + reg[:,y,x][2]*obs_/imean[y,x] + reg[:,y,x][4]*war_/warmean[y,x]
            mu=mu_clim/reg[:,y,x][0]*np.log1p(np.expm1(reg[:,y,x][0])*logarg)
            sigma=reg[:,y,x][3]*sigma_clim*np.sqrt(mu/mu_clim)        
            delta=delta_clim
            
            quants=delta+sp.stats.gamma.ppf(qVals,np.power(mu/sigma,2),scale=np.power(sigma,2)/mu,loc=0)
            quants[quants<0.]=0.
            simPrcp[:,:,y,x]=quants
                
                    
                        
    
    # -------- Apply precipitation detection threshold ------------------------
    # all precip less than 0.1 mm/hr is considered zero precip
    simPrcp[simPrcp<0.1] = 0.

    return(simPrcp)




