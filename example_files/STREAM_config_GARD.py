'''
Example usage of STREAM noise generation and precipitation simulation
In this example, STREAM is used to generate a correlated noise field that replicates the spatial correlation structure of an En-GARD output downscaled precipitation field.
A correlated noise field is generated for 1950-1955, and this noise field can then be used to post-process the En-GARD output field.

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
seed = 42    # seed for the random number generator used in random noise generation
ts = None    # timestep to run simulation at, in hours. Default is the timestep of the precipitation file (e.g. hourly for hourly precip data). Timestep cannot be smaller than precipitation data timestep. Leave this value as None if wanting to use the timestep of the input precipitation data.
n = 6*365       # number of timesteps to generate data for - in this case 6 years
dt = datetime(2050, 1, 1)  # start date

# path to the precipitation files - this is the raw output of the En-GARD downscaling model (before post-processing step), broken into yearly files to make it easier for users to download in this case
pcp_file = ('../example_data/gard_out_pcp2050.nc',
            '../example_data/gard_out_pcp2051.nc',
            '../example_data/gard_out_pcp2052.nc',
            '../example_data/gard_out_pcp2053.nc',
            '../example_data/gard_out_pcp2054.nc',
            '../example_data/gard_out_pcp2055.nc',
           )
pvar = 'pcp'  # variable name for field to replicate correlation from
# Optional: provide list of pcp filenames instead of single filename (e.g., multiple decadal files) 
# Precipitation file must have coordinates (time,lat,lon), and, if lat and lon are 2D variables, their dimensions must be (y,x)
# If precipitation file is not in correct format, this can easily be altered during optional preprocess_pcp function below

# In this example, advection is turned off because the time step is 1 day
advection = False  # whether to enable advection, generally not needed for time steps 6 hrs or greater 

wind_file = None # path to wind file, if using advection, else None
# Optional: provide list of wind filenames instead of single filename (e.g., multiple decadal files) 
uname = 'U700' # Variable name of east-west wind, if using advection, else None
vname = 'V700' # Variable name of north-south wind, if using advection, else None
# Wind file must have coordinates (time,lat,lon) and the lat/lon coordinates must have the same dimensions as the precip file (e.g. 1D for rectilinear grid or 2D for curvilinear grid)
# If wind file is not in correct format, this can easily be fixed during optional preprocess_wind function below
preprocess_wind = None   # Just set preprocess_wind to None if no preprocessing required


runtag = '_GARD'  # tag for the run, e.g. no_advection, errorModel1, etc. or set to '' for no tag
generateNoise = True # Set to true if generating correlated noise field from precipitation field
noiseFile = '../example_output/noise_output%s.nc'%runtag # This is the filename that the correlated noise will be saved to, if generateNoise = False, this should be an existing noise file that can be used in simulating precip fields

# Precipitation is False here because in this application we are only trying to generate a correlated noise field to use in GARD post processing
simulatePrecip = False # Set to true if using correlated noise field to simulate new precipitation field
# If simulatePrecip = True, user needs to define the CDF for precipitation at each grid cell and time step
# Define this in the distrCDF function below
paramsFile = None # File containing parameters that define distribution of precipitation, which will be used to simulate precipitation field from correlated noise
precipOutFile = None # File name for simulated precipitation output

verbose = True # Finally, set verbose to True if you want to see updates from STREAM as it runs

# -----------------------------------------------------------------------------------------------
# DEFINE PREPROCESSING FUNCTIONS AND PRECIP CDF

# Define any preprocessing that needs to be applied to pcp_file to get it in correct format with 1-D dimensions ('time', 'lat', 'lon'). See example functions below.

preprocess_pcp = None # If no preprocessing is required, set this to None

def preprocess_pcp(ds_name):
    '''
    # Function to rename dimensions of En-GARD output and change lat and lon to 1D fields since the grid is rectilinear
    '''
    ds = xr.open_mfdataset(ds_name)
    ds = ds.assign(y=ds.lat[:,0].values).assign(x=ds.lon[0,:].values-360).drop_vars(('lat','lon')).rename({'y':'lat','x':'lon'})
    return(ds)


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
    # Save this data to a netcdf, using compression to minimize file size
    correlated_noise.to_netcdf(noiseFile,encoding = {'noise': {"zlib": True, "complevel": 4}})

if simulatePrecip:
    # Generate precipitation fields based on correlated noise fields
    # Must include parameters to describe precipitation distribution at each grid cell and time step
    precip_sim = PrecipSimulation(obs_file=pcp_file,pvar=pvar,param_file=paramsFile,runtag=runtag)
    
    precip_out = precip_sim.simulatePrecip(dt=dt,ts=n,obsFile=pcp_file,noiseFile=noiseFile,
                                           paramsFile=paramsFile,cdf_func=distrCDF,corr=True,
                                          preprocess=preprocess_pcp)

    precip_out.to_netcdf(precipOutFile)








