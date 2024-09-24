
import numpy as np
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
import scipy as sp
import xarray as xr
import pandas as pd

# global area
# area = False

class PrecipSimulation:
    
    def __init__(self, pcp_file, pvar, param_file, runtag):
        self.obs_file = pcp_file
        self.pvar = pvar
        self.param_file = param_file
        self.runtag = runtag
        self.area_set = False

        global area_set
        area_set = False
    
    def set_area(self,ysize,xsize,dset=21):
        '''
        Author: Kaidi Peng
        '''
        
        # the default window size is 21*21
        global area
        area = np.zeros((ysize,xsize))
        d = dset
        area.fill((d)**2)
        rads= int((d-1)/2) # if d=21, rads=10
        
        a = np.reshape(np.arange((rads+1),d),(1,rads))
        b = np.reshape(np.arange((d-1),rads,-1),(1,rads))
        
        area[:rads,:rads] = np.transpose(a)*a
        area[(ysize-rads):,:rads] = np.transpose(b)*a
        area[:rads,(xsize-rads):] = np.transpose(a)*b
        area[(ysize-rads):,(xsize-rads):] = np.transpose(b)*b
        
        for i in range(rads+1,d):
            area[i-(rads+1),rads:(xsize-rads)] = i*d
            area[(ysize+rads)-i,rads:(xsize-rads)] = i*d
            area[rads:(ysize-rads),i-(rads+1)] = i*d
            area[rads:(ysize-rads),(xsize+rads)-i] = i*d
        
        global area_set
        area_set = True

    ## -------------------------------------------------------------------------
    
    def getWARfield(self,field,r):
        '''
        Function to retrieve wetted area ratio (WAR) across entire field
        '''
        ysize = np.shape(field)[0]
        xsize = np.shape(field)[1]
        
        if area_set == False:
            self.set_area(ysize,xsize)
        
        rainy = np.zeros(np.shape(field))
        # default rain threshold: 0.1mm/hr
        # when the observation > 0.1mm/hr, it is considered as rain
        rainy[field>=0.1] = 1 
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
    
    def distrCDF(self,obs_,noise):
        '''
        Function that calculates conditional censored shifted gamma distribution (CSGD)
        for a 3D (time, lat, lon) field of precipitation based on the precipitation values,
        wetted area ratio, noise field, and a set of regression parameters for the region.
        Authors: Daniel Wright, Sam Hartke, Kaidi Peng
        '''
        # Calculate Wetted Area Ratio
        war_ = np.zeros(np.shape(obs_))
        ts = np.shape(obs_)[0]
        for i in range(ts):
            war_[i,:,:] = self.getWARfield(obs_[i,:,:],10)
        
        params = xr.open_dataset(self.param_file) # Open params_file
        params = params.expand_dims(dim={"time": np.shape(obs_)[0]}).load() # Extend the original data along the time dimension of the obs
        
        logarg=params['par2'].values + params['par3'].values*obs_/params['mean'].values + params['par5'].values*war_/params['WARmean'].values
        mu=params['clim1'].values/params['par1'].values*np.log1p(np.expm1(params['par1'].values)*logarg)
        sigma=params['par4'].values*params['clim2'].values*np.sqrt(mu/params['clim1'].values)        
        delta=params['clim3'].values
        
        quants=delta+sp.stats.gamma.ppf(noise,np.power(mu/sigma,2),scale=np.power(sigma,2)/mu,loc=0)
        quants[quants<0.]=0.
        
        return(quants)
        
    # ----------------------------------------------------------------------------
    
    def simulatePrecip(self, dt, ts, noiseFile, cdf_func=None, corr=True,
                       verbose=False,pcp_thr=0.1,preprocess_pcp=None):
        '''
        Function to convert uniformly distributed noise into precipitation values
        Authors: Sam Hartke, Kaidi Peng
        '''
        end_dt = dt + timedelta(hours=(ts-1)) # end date of simulation
        if verbose: print("Generating precip ensemble for %s - %s"%(dt.strftime("%Y-%m-%d"),end_dt.strftime("%Y-%m-%d")))
        
        
        # Read in Precipitation Field  
        # Optional precipitation pre-processing step defined by user
        if preprocess_pcp!=None:
            if verbose: print('Preprocessing precip data.')
            ds = preprocess_pcp(self.obs_file)
        else:
            ds = xr.open_dataset(self.obs_file)
            
        obs = ds.sel(time=slice(dt.strftime('%Y-%m-%d'),end_dt.strftime('%Y-%m-%d'))).set_coords(('lat', 'lon'))[self.pvar] # grab IMERG data from simulation period
        obs = obs.where(obs>=pcp_thr).fillna(0.).transpose('time', 'lat', 'lon')
      
        if corr:
            # Open correlated noise file
            ds = xr.open_dataset(noiseFile).sel(time=slice(dt.strftime('%Y-%m-%d'),end_dt.strftime('%Y-%m-%d')))
            q = ds['noise'].values
        
        else:
            # Generate uncorrelated (white) noise
            q = np.random.uniform(size=(n_ens,ts,obs.shape[0],obs.shape[1])).astype('float16')
        
        # Set upper threshold for uniform noise so that unrealistically extreme
        # values don't get selected from conditional distribution tails
        q[q>0.995] = 0.995
        
        # Apply CDF converting noise values into precipitation values
        # This CDF can be conditional on not just noise values, but also precipitation values
        if cdf_func == None:
            simPrcp = self.distrCDF(obs, q)
        else:
            simPrcp = cdf_func(**locals())

        # all precip less than pcp_thr is considered zero precip
        simPrcp[simPrcp<pcp_thr] = 0.

        # Convert to xarray dataset with lat, lon values
        final_ds = obs.to_dataset().copy()
        final_ds['sim_prcp'] = (('time','lat','lon'),simPrcp)
        
        return(final_ds.drop(self.pvar))

