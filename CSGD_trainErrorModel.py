# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 16:25:02 2020

@author: Sam

Training IMERG error model with covariates over entire study area
using 2005-2007 data and WAR covariate

"""

import numpy as np
from netCDF4 import Dataset
from CSGD_utilities import fitcsgd_clim,fit_regression_simplified




#==============================================================================




# get hourly data over entire study area for 2006, 2007, and 2008

# -------------  retrieve IMERG dataset at this pixel  -----------------


ds = Dataset("csgd/data/IMERG2005.hourly.nc")

imerg = ds.variables['prcp'][:,:,24*60:(8760-61*24)] # grabbing data from all months except Jan, Feb, Nov, and Dec


ds = Dataset("csgd/data/IMERG2006.hourly.nc")

imerg = np.concatenate((ds.variables['prcp'][:,:,24*60:(8760-61*24)],imerg),axis=2)


ds = Dataset("csgd/data/IMERG2007.hourly.nc")

imerg = np.concatenate((ds.variables['prcp'][:,:,24*60:(8760-61*24)],imerg),axis=2)

imerg[imerg<0.1] = 0. # apply detection threshold at 0.1 mm/hr


ysize = imerg.shape[0]
xsize = imerg.shape[1]
zsize = imerg.shape[2]


# --------------  retrieve Stage IV dataset at this pixel  ----------------

ds = Dataset("csgd/data/NexradIV2005_0.1deg.hourly.nc")
    
stIV = ds.variables['prcp'][:,:,24*60:(8760-61*24)] # grabbing data from all months except Jan, Feb, Nov, and Dec


ds = Dataset("csgd/data/NexradIV2006_0.1deg.hourly.nc")
    
stIV = np.concatenate((ds.variables['prcp'][:,:,24*60:(8760-61*24)],stIV),axis=2)


ds = Dataset("csgd/data/NexradIV2007_0.1deg.hourly.nc")
    
stIV = np.concatenate((ds.variables['prcp'][:,:,24*60:(8760-61*24)],stIV),axis=2)

stIV[stIV<0.1] = 0.




# ----- retrieve wetted area ratio (WAR) covariate ------
# ----- these files are created in WARanalysis.py  ------

ds = Dataset("csgd/data/WetAR2005.r10.hourly.nc")

WAR = ds.variables['war'][:,:,24*60:(8760-61*24)] # grabbing data from all months except Jan, Feb, Nov, and Dec


ds = Dataset("csgd/data/WetAR2006.r10.hourly.nc")

WAR = np.concatenate((ds.variables['war'][:,:,24*60:(8760-61*24)],WAR),axis=2)


ds = Dataset("csgd/data/WetAR2007.r10.hourly.nc")

WAR = np.concatenate((ds.variables['war'][:,:,24*60:(8760-61*24)],WAR),axis=2)


ds=None




# =============================================================================
# ===========   train IMERG error model at each pixel in study area    ========
# 

# indicate whether to train linear or nonlinear regression within CSGD model framework
lin = False

pars = np.empty((10,ysize,xsize)) # array to hold calibrated error model parameters

# window size over which to "regionalize" pixel-scale error model
w = 5 # must be odd
wh = int((w-1)/2) # half of window size

for y in np.arange(wh,ysize,w):
    
    if y%10==0:
        print(y)
    
    for x in np.arange(wh,xsize,w):
        
        y1 = np.max((0,y-wh)) # starting y index of subset window
        y2 = np.min((ysize,y+wh+1)) # ending y index of subset window
        x1 = np.max((0,x-wh)) # starting x index of subset window
        x2 = np.min((xsize,x+wh+1)) # ending x index of subset window
        
        # train each error model on pixel + 2 surrounding pixels in every direction (25 total pixels)
        stIV_ = stIV[y1:y2,x1:x2,:].flatten()
        imerg_ = imerg[y1:y2,x1:x2,:].flatten()
        WAR_ = WAR[y1:y2,x1:x2,:].flatten()
        
        
        extraCovar = np.ones(len(WAR_))
        covar_train = np.column_stack((WAR_,extraCovar,extraCovar)) # combine these covariates
    
        
        # train climatological CSGD
        clim_params = fitcsgd_clim(imerg_).x
        
        # store in parameter array
        pars[0,y1:y2,x1:x2] = clim_params[0]
        pars[1,y1:y2,x1:x2] = clim_params[1]
        pars[2,y1:y2,x1:x2] = clim_params[2]

        # train  error models with covariate
        reg_covar = fit_regression_simplified(imerg_, stIV_, clim_params,
                                                linear = lin, include_covariates=True, 
                                                covars=WAR_, whichcovars=[1])
        
        # store in parameter array
        pars[3,y1:y2,x1:x2] = reg_covar.x[0]
        pars[4,y1:y2,x1:x2] = reg_covar.x[1]
        pars[5,y1:y2,x1:x2] = reg_covar.x[2]
        pars[6,y1:y2,x1:x2] = reg_covar.x[3]
        pars[7,y1:y2,x1:x2] = reg_covar.x[4]
        
        if pars[3,y,x] == 0.500:
            print(y,x,pars[3:8,y,x])
        
        # store imerg mean in parameter array
        pars[8,y1:y2,x1:x2] = np.mean(imerg_)
        pars[9,y1:y2,x1:x2] = np.mean(WAR_)
        



# save parameter grid to netcdf

if lin==True:
    model="L"
else:
    model="NL"
    

fname = "csgd/%smodel_covars.nc"%model


print('Writing %s \n'%fname)

new_cdf = Dataset(fname, 'w', format = "NETCDF4", clobber=True)

# create dimensions
new_cdf.createDimension('lat', 100)
new_cdf.createDimension('lon', 150)


# add lat, and lon variables
latitude = new_cdf.createVariable('latitude', 'f8', ('lat'), zlib=True)
latitude.units = 'degrees_north'
latitude.long_name = 'latitude'
latitude[:] = np.arange(44.95,35.05,-0.1)

longitude = new_cdf.createVariable('longitude', 'f8', ('lon'), zlib=True)
longitude.units = 'degrees_east'
longitude.long_name = 'longitude'
longitude[:] = np.arange(-99.95,-85.05,0.1)


climpar1 = new_cdf.createVariable('clim1', 'f8', ('lat','lon'), zlib=True)
climpar1.units = '--'
climpar1.long_name = 'Climatological mu'
climpar1[:,:] = pars[0,:,:]


climpar2 = new_cdf.createVariable('clim2', 'f8', ('lat','lon'), zlib=True)
climpar2.units = '--'
climpar2.long_name = 'Climatological sigma'
climpar2[:,:] = pars[1,:,:]


climpar3 = new_cdf.createVariable('clim3', 'f8', ('lat','lon'), zlib=True)
climpar3.units = '--'
climpar3.long_name = 'Climatological delta'
climpar3[:,:] = pars[2,:,:]


par1 = new_cdf.createVariable('par1', 'f8', ('lat','lon'), zlib=True)
par1.units = '--'
par1.long_name = 'Regression par 1'
par1[:,:] = pars[3,:,:]


par2 = new_cdf.createVariable('par2', 'f8', ('lat','lon'), zlib=True)
par2.units = '--'
par2.long_name = 'Regression par 2'
par2[:,:] = pars[4,:,:]


par3 = new_cdf.createVariable('par3', 'f8', ('lat','lon'), zlib=True)
par3.units = '--'
par3.long_name = 'Regression par 3'
par3[:,:] = pars[5,:,:]


par4 = new_cdf.createVariable('par4', 'f8', ('lat','lon'), zlib=True)
par4.units = '--'
par4.long_name = 'Regression par 4'
par4[:,:] = pars[6,:,:]


par5 = new_cdf.createVariable('par5', 'f8', ('lat','lon'), zlib=True)
par5.units = '--'
par5.long_name = 'Covariate WAR regression par 1'
par5[:,:] = pars[7,:,:]



imean = new_cdf.createVariable('mean', 'f8', ('lat','lon'), zlib=True)
imean.units = '--'
imean.long_name = 'IMERG mean'
imean[:,:] = pars[8,:,:]


WARmean = new_cdf.createVariable('WARmean', 'f8', ('lat','lon'), zlib=True)
WARmean.units = '--'
WARmean.long_name = 'WAR mean'
WARmean[:,:] = pars[9,:,:]



new_cdf.close()





