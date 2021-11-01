# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:13:43 2020

@author: Sam Hartke

Script to generate and save netcdf of ensemble of correlated uniform noise using
pysteps replication of local spatial structures and semi-lagrangian scheme
based on MERRA2 wind fields

"""

from netCDF4 import Dataset, date2num, num2date
import numpy as np
from datetime import date, timedelta, datetime
import math
import scipy as sp
from tqdm import tqdm
from pysteps.noise.fftgenerators import initialize_nonparam_2d_ssft_filter
from pysteps.noise.fftgenerators import generate_noise_2d_ssft_filter
from scipy.stats import pearsonr

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

#==============================================================================

# function to calculate lag-1 temporal correlation between most recent precip fields
def getAlpha(data0,data1):
    
    # get average temporal correlation of field at timestep t
    lp, _ = pearsonr(data1.flatten(),data0.flatten())
    
    return(lp)



#==============================================================================
    
def getCorrNoise(n, h, dt,obsFile,windFile):
    
    ysize=100
    xsize=150
    
    
    # ---- grab hourly MERRA2 wind data at 850 mb over 35N-45N, 85W-100W [m/s]
    # MERRA2 files were downloaded from GES-DISC and aggregated to yearly files in writeWindNetcdfs.py
    ds = Dataset(windFile)
    
    # get number of hours between start of file and date dt at hour h
    d_start = num2date(ds.variables['time'][0],ds.variables['time'].units)
    nd = 24*(dt - date(d_start.year,d_start.month,d_start.day)).days + h
    
    u_values = ds.variables['U850'][:,:,nd:(nd+n)] # eastward wind [m/s]
    v_values = ds.variables['V850'][:,:,nd:(nd+n)] # northward wind [m/s]
    
    
    # ----  open hourly IMERG data  ----
    ds = Dataset(obsFile)
    
    # get number of hours between start of file and date dt at hour h
    d_start = num2date(ds.variables['time'][0],ds.variables['time'].units)
    nd = 24*(dt - date(d_start.year,d_start.month,d_start.day)).days + h
    
    imerg = ds.variables['prcp'][:,:,nd:(nd+n)].astype('float32')
    imerg[imerg<0.1] = 0.
    ds=None
    
    # grid spacing [m] - 0.1 degree resolution corresponds to ~11 km or ~11000 m
    dx = 11000
    dy = 11000
    
    # create array to hold simulated noise
    s = np.empty((n,ysize,xsize),dtype=np.float32)
    
    # --- SIMULATE FIRST FIELD ---
    if len(imerg[:,:,0][imerg[:,:,0]>=0.1]) > 750:
        # only use pysteps smoothing filter if there is actually rain in the study area
        Fnp = initialize_nonparam_2d_ssft_filter(imerg[:,:,0]) 
        s[0,:,:] = generate_noise_2d_ssft_filter(Fnp)
        firstRain=True
        
    else:
        # otherwise just use smoothed white noise
        s[0,:,:] = np.random.normal(size=(ysize,xsize))
        firstRain=False
    
    
    last_rn = s[0,:,:] # record this instance of random noise as most recent instance of random noise
    
    
    # set initial value of alpha parameter - determines degree to which noise is incorporated into field at each timestep
    alpha = 0.95
    
    # --- ADVECT FIELD FORWARD IN TIME ---
    for hr in range(1,n):
        
        
        # get wind values from PREVIOUS hour and calculate dx, dy that correspond to this time step
        u = u_values[:,:,hr-1] # eastward wind [m/s]
        
        # use cv.resize function to regrid u wind to 0.1 degree
        deltax = u*60*60 # distance wind travels each time step in positive x dir [m]
        dix = deltax//dx # number of grid steps to shift in semi-lagrangian scheme in units of [0.1 deg pixels]
        
        
        v = v_values[:,:,hr-1] # northward wind [m/s]
        deltay = -v*60*60 # distance wind travels each time step in positive y dir [m]
        diy = deltay//dy # number of grid steps to shift in semi-lagrangian scheme [0.1 deg pixels]
        imergh = imerg[:,:,hr]
        
        
        # use pySTEPS to generate noise with power spectrum of IMERG field
        # only use pysteps smoothing filter if there is actually rain in the study area
        
        if len(imergh[imergh>=0.1]) > 750: # atleast 5% of study area must be rainy
            
            Fnp = initialize_nonparam_2d_ssft_filter(imergh,win_size=(64,64))
            rn = generate_noise_2d_ssft_filter(Fnp)
            
            # get average temporal correlation of field in previous timestep
            alpha = getAlpha(imergh,imerg[:,:,hr-1])
           
            firstRain=True
        
        
        else:
            
            # if field does not have at least 5% rainy pixels, generate random field 
            # using spatial correlation structure of last instance of rainfall 
            
            if firstRain==True:
                
                rn = generate_noise_2d_ssft_filter(Fnp)
            
            else:
                # if first instance of a "rainy field" hasn't occurred yet in study period, use white noise
                rn = np.random.normal(size=(ysize,xsize)) # white noise
        
        
        if np.isnan(rn[0,0])==True: # if pysteps smoothing process doesn't work for some reason, use last realization of smoothed noise
            
            rn = last_rn
            
        else:
            
            last_rn = rn
        
        
        # -- advect noise values from previous time step & perturb with noise field rn --
        # move through each row and find previous noise values that would have
        # advected to that row
        for k in range(0,ysize):
            
            kbefore = ((k-diy[k,:])%ysize).astype('int')
            
            lbefores = ((np.arange(xsize)-dix[k,:])%xsize).astype('int') # horizontal indices
            
            s[hr,k,:] = alpha*s[hr-1,kbefore,lbefores] + np.sqrt(1.-alpha**2)*rn[k,:]

    
    
    
    # return noise field
    return(s)


#==============================================================================
#==============================================================================

def generateNoise(n_ens,ts,dt,obsFile,windFile,newFile):
    
    hr = 0 # hour to start simulation at - we assume this is 0, but this can be changed    
    
    end_dt = dt + timedelta(hours=(hr+ts-1)) # end date of simulation
    
    
    #print("Generating %d-member noise ensemble for %s to %s"%(n_ens,dt.strftime("%Y-%m-%d"),end_dt.strftime("%Y-%m-%d")))
    
    
    
    save = True
    
    if save == True:
        
        # save noise to netcdf
        
        new_cdf = Dataset(newFile, 'w', format = "NETCDF4", clobber=True)
        
        # create array of time stamps
        time_hrs = [datetime(dt.year,dt.month,dt.day,0,0)+n*timedelta(hours=1) for n in range(ts)]
        units = 'hours since 1970-01-01 00:00:00 UTC'
        
        # open input precipitation netcdf to get latitude and longitude arrays
        ds = Dataset(obsFile)
        ysize = len(ds['latitude'][:])
        xsize = len(ds['longitude'][:])
        
        # create dimensions
        new_cdf.createDimension('lat', ysize)
        new_cdf.createDimension('lon', xsize)
        new_cdf.createDimension('ens_n', n_ens)
        new_cdf.createDimension('time', ts)
        
        # write time stamps to variable
        time = new_cdf.createVariable('time','d', ('time'))
        time.units = units
        time[:] = date2num(time_hrs,units,calendar="gregorian")
        
        # add lat, and lon variables
        latitude = new_cdf.createVariable('latitude', 'f4', ('lat'), zlib=True,least_significant_digit=2)
        latitude.units = 'degrees_north'
        latitude.long_name = 'latitude'
        
        latitude[:] = ds['latitude'][:]
        
        longitude = new_cdf.createVariable('longitude', 'f4', ('lon'), zlib=True,least_significant_digit=2)
        longitude.units = 'degrees_east'
        longitude.long_name = 'longitude'
        longitude[:] = ds['longitude'][:]

        ds = None
        
        noise = new_cdf.createVariable('q', 'f4', ('ens_n','time','lat','lon'), zlib=True,least_significant_digit=4)
        noise.units = '--'
        noise.long_name = 'Uniform Noise'
        
        
        for n in tqdm(range(0,n_ens),desc='Generating %d-member noise ensemble'%n_ens):
            
            # retrieve instance of correlated white noise for ts timesteps starting at date dt and hour hr
            s = getCorrNoise(ts,hr,dt,obsFile,windFile)
            
            s = 0.5*(1+sp.special.erf((s/math.sqrt(2)))) # convert to uniform noise
            
            noise[n,:,:,:] = s # store in netcdf noise array
            
        
        new_cdf.close()




