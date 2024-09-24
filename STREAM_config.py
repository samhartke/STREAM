'''
Example usage of STREAM noise generation and precipitation simulation
In this example, STREAM is used to generate a correlated noise field that replicates the spatial correlation structure of an IMERG precipitation field.
A correlated noise field is generated for Apr 1 - Apr 8, 2013, and this noise field is then used to simulate a possible "true" precipitation field based on IMERG
and a trained censored shifted gamma distribution error model.
'''

from datetime import datetime
from STREAM_PrecipSimulation import PrecipSimulation # Import necessary modules from the STREAM_Utils.py file
from STREAM_Utils import NoiseGenerator, Validate
import scipy as sp
import numpy as np
import xarray as xr

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# START OF USER INPUT

# Define parameters and file names

wsize = (128,128)  # window size for the noise generation
seed = 1    # seed for the random number generator
ts = None    # timestep to run simulation at, in hours. Default is the timestep of the precipitation file (e.g. hourly for hourly precip data). Timestep cannot be smaller than precipitation data timestep.
n = 24*7       # number of timesteps to generate data for
dt = datetime(2000, 1, 1)  # start date
pcp_file = 'path/to/precip/file.nc'  # path to the precipitation file
pvar = 'prcp'  # variable name for field to replicate correlation from
# Optional: provide list of pcp filenames instead of single filename (e.g., multiple decadal files) 
# Precipitation file must have coordinates (time,lat,lon), and, if lat and lon are 2D variables, their dimensions must be (y,x)
# If precipitation file is not in correct format, this can easily be altered during optional preprocess_pcp function below


advection = True  # whether to enable advection, generally not needed for time steps 6 hrs or greater

wind_file = 'path/to/wind/file.nc'  # path to wind file, if using advection, else None
# Optional: provide list of wind filenames instead of single filename (e.g., multiple decadal files) 
uname = 'U700' # Variable name of east-west wind, if using advection, else None
vname = 'V700' # Variable name of north-south wind, if using advection, else None
# Wind file must have coordinates (time,lat,lon) and the lat/lon coordinates must have the same dimensions as the precip file (e.g. 1D for rectilinear grid or 2D for curvilinear grid)
# If wind file is not in correct format, this can easily be fixed during optional preprocess_wind function below
preprocess_wind = None   # Just set preprocess_wind to None if no preprocessing required


runtag = '_descriptive_runtag'  # tag for the run, e.g. no_advection, errorModel1, etc. or set to '' for no tag
generateNoise = True # Set to true if generating correlated noise field from precipitation field
noiseFile = 'noise_output%s.nc'%runtag # This is the filename that the correlated noise will be saved to, if generateNoise = False, this should be an existing noise file that can be used in simulating precip fields

simulatePrecip = True # Set to true if using correlated noise field to simulate new precipitation field
# If simulatePrecip = True, user needs to define the CDF for precipitation at each grid cell and time step
# Define this in the distrCDF function below
paramsFile = 'path/to/params/file.nc' # File containing parameters that define distribution of precipitation, which will be used to simulate precipitation field from correlated noise
precipOutFile = 'path/to/precip/output.nc' # File name for simulated precipitation output

verbose = True # Finally, set verbose to True if you want to see updates from STREAM as it runs

# -----------------------------------------------------------------------------------------------
# DEFINE PREPROCESSING FUNCTIONS AND PRECIP CDF

# Define any preprocessing that needs to be applied to pcp_file to get it in correct format with 1-D dimensions ('time', 'lat', 'lon'). See example functions below.

preprocess_pcp = None # If no preprocessing is required, set this to None and delete preprocess_pcp function block below

def preprocess_pcp(precip_file_name):
    '''
    Function to preprocess input precipitation dataset to match required coordinates (time, lat, lon), etc.
    '''
    ds = xr.open_dataset(precip_file_name)
    
    return(ds)

# If transforming noise into precip field using simulatePrecip = True, define the cumulative distribution function (CDF)
# The user can define which input variables are required, but one of the inputs must be a 3D noise field, noise, with coordinates (time, lat, lon), and the output must return a 3D precip field
if simulatePrecip:
    
    def distrCDF(define_inputs_here):
        '''
        Define function to transform uniform noise field values into precipitation values
        '''
        
        return(return_pcp)


if generateNoise and advection:
    # Define any preprocessing that needs to be applied to wind_file to get it in correct format with 1-D dimensions ('time', 'lat', 'lon'). See example functions below.

    preprocess_wind = None   # Just set preprocess_wind to None if no preprocessing required
    
    def preprocess_wind(wind_file_name):
        '''
        Function to preprocess wind dataset to match required coordinates (time, lat, lon), etc.
        '''
        ds = xr.open_dataset(wind_file_name)
        
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
    precip_sim = PrecipSimulation(obs_file=pcp_file,pvar=pvar,param_file=paramsFile,runtag=runtag)
    
    precip_out = precip_sim.simulatePrecip(dt=dt,ts=n,obsFile=pcp_file,noiseFile=noiseFile,
                                           paramsFile=paramsFile,cdf_func=distrCDF,corr=True,
                                          preprocess=preprocess_pcp)

    precip_out.to_netcdf(precipOutFile)








