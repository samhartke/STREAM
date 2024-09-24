'''
Example usage of STREAM noise generation and precipitation simulation
In this example, STREAM is used to generate a correlated noise field that replicates the spatial correlation structure of an IMERG precipitation field.
A correlated noise field is generated for Apr 1 - Apr 8, 2013, and this noise field is then used to simulate a possible "true" precipitation field based on IMERG
and a trained censored shifted gamma distribution error model.
'''
import sys
import os
from datetime import datetime
import scipy as sp
import numpy as np
import xarray as xr

# Add src/ directory to the system path
sys.path.append(os.path.abspath("../src/"))
from STREAM_PrecipSimulation import PrecipSimulation # Import necessary modules from the STREAM_Utils.py file
from STREAM_Utils import NoiseGenerator, Validate

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# START OF USER INPUT

# Define parameters and file names

wsize = (128,128)  # window size for the noise generation
seed = 42    # seed for the random number generator
ts = None    # timestep to run simulation at, in hours. Default is the timestep of the precipitation file (e.g. hourly for hourly precip data). Timestep cannot be smaller than precipitation data timestep.
n = 24*10       # number of timesteps to generate data for - 10 days of hourly data in this case
dt = datetime(2013, 4, 1)  # start date
pcp_file = '../example_data/IMERG2013.hourly.nc'  # path to the precipitation file
pvar = 'prcp'  # variable name for field to replicate correlation from
# Precipitation file must have coordinates (time,lat,lon), and, if lat and lon are 2D variables, their dimensions must be (y,x)
# If precipitation file is not in correct format, this can easily be altered during optional preprocess_pcp function below


advection = True  # use advection since we're running STREAM at an hourly timestep

wind_file = '../example_data/ERA5_CONUS_daily_2013.nc'  # path to ERA5 daily dataset
uname = 'U700' # Variable name of east-west wind, if using advection, else None
vname = 'V700' # Variable name of north-south wind, if using advection, else None
# Wind file must have coordinates (time,lat,lon) and the lat/lon coordinates must have the same dimensions as the precip file (e.g. 1D for rectilinear grid or 2D for curvilinear grid)
# If wind file is not in correct format, this can easily be fixed during optional preprocess_wind function below
preprocess_wind = None   # Just set preprocess_wind to None if no preprocessing required


runtag = '_IMERG_CSGD'  # tag for the run, e.g. no_advection, errorModel1, etc. or set to '' for no tag
generateNoise = True  # Set to true if generating correlated noise field from precipitation field
noiseFile = '../example_output/noise_output%s.nc'%runtag  # This is the filename that the correlated noise will be saved to, if generateNoise = False, this should be an existing noise file that can be used in simulating precip fields

simulatePrecip = True  # Set to true if using correlated noise field to simulate new precipitation field
# If simulatePrecip = True, user needs to define the CDF for precipitation at each grid cell and time step
# Define this in the distrCDF function below
param_file = '../example_data/NLmodel_covars.nc'  # File containing parameters that define distribution of precipitation, which will be used to simulate precipitation field from correlated noise
precipOutFile = '../example_output/precip_output%s.nc'%runtag  # File name for simulated precipitation output

verbose = True # Finally, set verbose to True if you want to see updates from STREAM as it runs

# -----------------------------------------------------------------------------------------------
# DEFINE PREPROCESSING FUNCTIONS AND PRECIP CDF

# Define any preprocessing that needs to be applied to pcp_file to get it in correct format with 1-D dimensions ('time', 'lat', 'lon'). See example functions below.

preprocess_pcp = None # If no preprocessing is required, set this to None and delete preprocess_pcp function block below

def preprocess_pcp(ds_name):
    '''
    # Function to rename latitude and longitude variables to lat and lon and set them as coordinates
    '''
    ds = xr.open_dataset(ds_name)
    ds = ds.rename({'latitude':'lat','longitude':'lon'}).set_coords(('lat','lon'))
    
    return(ds)

# If transforming noise into precip field using simulatePrecip = True, define the cumulative distribution function (CDF)
# The user can define which input variables are required, but one of the inputs must be a 3D noise field, noise, with dimensions (time, y, x), and the output must return a 3D precip field
if simulatePrecip:

    def distrCDF(obs_,noise,params_file):
        '''
        Function that calculates conditional censored shifted gamma distribution (CSGD)
        for a 3D (time, lat, lon) field of precipitation based on the precipitation values,
        wetted area ratio, noise field, and a set of regression parameters for the region.
        Authors: Daniel Wright, Sam Hartke, Kaidi Peng
        
        This is currently the default CDF used and defined in STREAM_PrecipSimulation.py,
        but it is included here to show users how one would include their own specific CDF.
        '''
        # Calculate Wetted Area Ratio
        war_ = np.zeros(np.shape(obs_))
        ts = np.shape(obs_)[0]

        precip_sim = PrecipSimulation(**locals())
        for i in range(ts):
            war_[i,:,:] = precip_sim.getWARfield(obs_[i,:,:],10)
        
        params = xr.open_dataset(params_file) # Open params_file
        params = params.expand_dims(dim={"time": np.shape(obs_)[0]}).load() # Extend the original data along the time dimension of the obs
        
        logarg=params['par2'].values + params['par3'].values*obs_/params['mean'].values + params['par5'].values*war_/params['WARmean'].values
        mu=params['clim1'].values/params['par1'].values*np.log1p(np.expm1(params['par1'].values)*logarg)
        sigma=params['par4'].values*params['clim2'].values*np.sqrt(mu/params['clim1'].values)        
        delta=params['clim3'].values
        
        pcp_out=delta+sp.stats.gamma.ppf(noise,np.power(mu/sigma,2),scale=np.power(sigma,2)/mu,loc=0)
        pcp_out[pcp_out<0.]=0.
        
        return(pcp_out)


if generateNoise and advection:
    # Define any preprocessing that needs to be applied to wind_file to get it in correct format with 1-D dimensions ('time', 'lat', 'lon'). See example functions below.
    
    def preprocess_wind(ds_name):
        '''
        # Function to rename latitude and longitude variables to lat and lon
        '''
        
        ds = xr.open_dataset(ds_name).rename({'latitude':'lat','longitude':'lon'})
        
        return(ds)

else:
    preprocess_wind = None

# END OF USER INPUT
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

if generateNoise:
    # Initialize the NoiseGenerator with the window size and seed
    noise_gen = NoiseGenerator(pcp_file=pcp_file, dt=dt, pvar=pvar, preprocess_pcp=preprocess_pcp,
    preprocess_wind=preprocess_wind, wind_file=wind_file,uname=uname, vname=vname, wsize=wsize,
                               seed=seed, advection=advection,verbose=verbose)
    #noise_gen = NoiseGenerator(**locals())
    
    # Generate correlated noise with advection
    correlated_noise = noise_gen.get_corr_noise(
        n=n,
        dt=dt,
        seednum=seed,
        runtag=runtag,
        ts = ts,
    )
    
    # correlated_noise is now a correlated noise based on input precip fields and advection of input wind fields
    # Save this data to a netcdf
    correlated_noise.to_netcdf(noiseFile)

if simulatePrecip:
    # Generate precipitation fields based on correlated noise fields
    # Must include parameters to describe precipitation distribution at each grid cell and time step
    precip_sim = PrecipSimulation(pcp_file=pcp_file,pvar=pvar,param_file=param_file,runtag=runtag)
    
    precip_out = precip_sim.simulatePrecip(dt=dt, ts=n, noiseFile=noiseFile,corr=True,#cdf_func=distrCDF,
                                          preprocess_pcp=preprocess_pcp)

    precip_out.to_netcdf(precipOutFile)



