# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:49:40 2021

@author: Sam Hartke
"""

import os
from datetime import date, timedelta, datetime
from STREAM_PrecipSimulation import simulatePrecip
from STREAM_PrecipSimulation_DBW import simulatePrecip_DBW
from STREAM_NoiseGeneration import generateNoise
from netCDF4 import Dataset,date2num
import numpy as np


# -----------------------------------------------------------------------------
# -------------------  INPUT PARAMETERS SECTION  ------------------------------


dt = date(2013,6,1) # date to start simulation at

nEns = 8 # number of ensemble members to generate

ts = 60*24 # number of timesteps to run simulation for [hrs]




# ---  input file names  ---

wd = "C:/Users/samia/OneDrive/QuantileWork/STREAMcode/"

obsInFname = wd + "IMERG2013.hourly.nc"  # path to satellite precipitation netcdf

windInFname = wd + "MERRA2_0.1.2013.hourly.nc"  # path to wind speed netcdf

paramsInFname = wd + "NLmodel_covars.nc"  # path to error model parameter netcdf




# ---  output file names  ---

end_dt = dt + timedelta(hours=(ts-1)) # end date of simulation
noiseOutFname = wd + "noise_%s_%s.nc"%(dt.strftime('%Y%m%d'),end_dt.strftime('%Y%m%d'))  # path for noise ensemble output

precipOutFname = wd + "STREAM_%s_%s.nc"%(dt.strftime('%Y%m%d'),end_dt.strftime('%Y%m%d'))  # path for precipitation ensemble output



# --- END OF INPUT PARAMETERS FOR STREAM  -------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#%%


# --- Generate noise ensemble and save to netcdf noiseOutFname
generateNoise(nEns,ts,dt,obsInFname,windInFname,noiseOutFname)

#%%
# --- Simulate STREAM ensemble of precipitation
simPrcp = simulatePrecip_DBW(dt,nEns,ts,obsInFname,noiseOutFname,paramsInFname)



# ----- Save simulated ensemble of precip to netcdf precipOutFname ------------

ysize = np.shape(simPrcp)[2]
xsize = np.shape(simPrcp)[3]

new_cdf = Dataset(precipOutFname, 'w', format = "NETCDF4", clobber=True)

# create array of time stamps
time_hrs = [datetime(dt.year,dt.month,dt.day,0,0)+n*timedelta(hours=1) for n in range(ts)]
units = 'hours since 1970-01-01 00:00:00 UTC'

# create dimensions
new_cdf.createDimension('lat', ysize)
new_cdf.createDimension('lon', xsize)
new_cdf.createDimension('ens_n', nEns)
new_cdf.createDimension('time', ts)

# write time stamps to variable
time = new_cdf.createVariable('time','d', ('time'))
time.units = units
time[:] = date2num(time_hrs,units,calendar="gregorian")

# add lat, and lon variables
latitude = new_cdf.createVariable('latitude', 'f4', ('lat'), zlib=True,least_significant_digit=2)
latitude.units = 'degrees_north'
latitude.long_name = 'latitude'
# open input precipitation netcdf to get latitude and longitude arrays
ds = Dataset(obsInFname)
latitude[:] = ds['latitude'][:]

longitude = new_cdf.createVariable('longitude', 'f4', ('lon'), zlib=True,least_significant_digit=2)
longitude.units = 'degrees_east'
longitude.long_name = 'longitude'
longitude[:] = ds['longitude'][:]

ds = None

prcp = new_cdf.createVariable('prcp', 'f4', ('ens_n','time','lat','lon'), zlib=True,least_significant_digit=3)
prcp.units = 'mm/hr'
prcp.long_name = 'Simulated Precipitation'
prcp[:,:,:,:] = simPrcp

new_cdf.close()



