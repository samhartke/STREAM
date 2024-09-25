
import numpy as np
from datetime import date, timedelta, datetime
import scipy as sp
from scipy.stats import pearsonr
import xarray as xr
import time
import random
import pandas as pd
import math
import xesmf as xe

class Utils:
    
    @staticmethod
    def latlon_distance(lat1, lon1, lat2, lon2):
        '''
        Code to compute the distance between two (lat,lon) points in meters
        Author: Daniel Wright
        '''
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat / 2.0) ** 2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
             np.sin(dlon / 2.0) ** 2)
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c * 1000.0

class Validate:

    def __init__(self, instance_NoiseGenerator):
        self.NoiseGen = instance_NoiseGenerator

    
    def checkCoords(self,ds):

        # Check if all required coordinates exist
        for coord in ("time","lat","lon"):
            if coord not in ds.coords:
                print(f"Coordinate '{coord}' is missing.")
                return False

        # If dataset lat and lon are 2D variables, ensure that dimensions are labeled as 'y' and 'x'
        if (len(ds['lat'].dims)>1) and (ds['lat'].dims != ('y', 'x')):
            print("Coordinate 'lat' does not have dimensions ('y', 'x').")
            return False
        
        if (len(ds['lon'].dims)>1) and (ds['lon'].dims != ('y', 'x')):
            print("Coordinate 'lon' does not have dimensions ('y', 'x').")
            return False
            
        if self.NoiseGen.verbose: print("Dataset has required coordinates.")
            
        return(True)

    
    def check_for_nan_and_inf(self,ds):
            
        for var_name, da in ds.data_vars.items():
            if da.isnull().any():
                print(f"Variable '{var_name}' contains NaN values. Filling in with 0.")
                    # Replace NaN values with 0
            ds[var_name] = da.fillna(0)
            
            if np.isinf(da.values).any():
                print(f"Variable '{var_name}' contains Inf values. Filling in with 0.")
                # Replace Inf values with 0
                ds[var_name] = xr.DataArray(np.where(np.isinf(da.values), 0, da.values), 
                                            dims=da.dims, coords=da.coords)
        
        return(ds)

    
    def checkWindField(self, wind_ds, precip_ds):
        '''
        Function to check that wind field files cover the same time steps and resolution as the precipitation file
        '''
        
        if self.checkCoords(wind_ds)==False:
            print("Dataset does not have correct coordinates: time, lat, lon.")
            return(None)

        wind_ds = self.check_for_nan_and_inf(wind_ds) # Fill in missing values, e.g. nans and infs, with 0.0

        # wind_ds = wind_ds.reindex(lat=wind_ds['lat'].sortby('lat'), lon=wind_ds['lon'].sortby('lon', ascending=False))
        if wind_ds['lat'].dims == ('y', 'x'): # ensure that dimensions are in correct order, whether using 1D or 2D data
            wind_ds = wind_ds.transpose('time','y','x')
        else:
            wind_ds = wind_ds.transpose('time','lat','lon')

        ## --- Check that assigned wind variable names, uname and vname, exist in dataset
        if self.NoiseGen.uname not in wind_ds.data_vars:
            print('Wind dataset does not contain variable %s.'%self.NoiseGen.uname)

        if self.NoiseGen.vname not in wind_ds.data_vars:
            print('Wind dataset does not contain variable %s.'%self.NoiseGen.vname)

        wind_ds = wind_ds[[self.NoiseGen.uname,self.NoiseGen.vname]]

        ## --- Check that wind and precip file cover the same area
        # Check if the latitude and longitude coordinates are the same
        lat1, lon1 = precip_ds['lat'], precip_ds['lon']
        lat2, lon2 = wind_ds['lat'], wind_ds['lon']
        
        if (not lat1.equals(lat2)) or (not lon1.equals(lon2)):
            print('Wind and precipitation files are not on the same grid. Regridding before running STREAM.')
            # Regrid to resolution of precipitation dataset if not equal
            regridder = xe.Regridder(wind_ds, precip_ds, method='bilinear')
            wind_ds = regridder(wind_ds)
        
        ## --- Check that wind file contains all dates present in precip file
        if not np.issubdtype(wind_ds['time'].dtype, np.datetime64):
            wind_ds['time'] = wind_ds.indexes['time'].to_datetimeindex()
        
        if not np.issubdtype(precip_ds['time'].dtype, np.datetime64):
            precip_ds['time'] = precip_ds.indexes['time'].to_datetimeindex()
        
        precip_time = precip_ds['time'].values
        wind_time = wind_ds['time'].values
        all_dates_present = np.isin(precip_time, wind_time).all()
        
        if not all_dates_present:
            print("Some dates in the precipitation data are missing in the wind data.")
            missing_dates = precip_time[~np.isin(precip_time, wind_time)]
            print("Missing dates:", missing_dates)  # Print the missing dates
            print('Automatically interpolating wind dataset to fill in missing time steps.')
            wind_ds = wind_ds.interp(time=precip_time)

        wind_ds = wind_ds.interp(time=precip_ds['time'].values)

        return(wind_ds)

    
    def checkPrecipField(self, prcp_ds, pvar, remove_duplicates=False):
        '''
        Function to check for duplicate times, missing data, etc. in precipitation data
        '''

        if self.checkCoords(prcp_ds)==False:
            print("Dataset does not have correct coordinates: time, lat, lon.")

        prcp_ds = self.check_for_nan_and_inf(prcp_ds) # Fill in missing values, e.g. nans and infs, with 0.0
        
        if prcp_ds['lat'].dims == ('y', 'x'): # ensure that dimensions are in correct order, whether using 1D or 2D data
            prcp_ds = prcp_ds.transpose('time','y','x')
        else:
            prcp_ds = prcp_ds.transpose('time','lat','lon')

        ## Check that assigned precipitation variable name, pvar, exists in dataset
        if pvar not in prcp_ds.data_vars:
            print('Precipitation dataset does not contain variable %s.'%pvar)

        ## Check if duplicate dates are found in precipitation data & remove them
        time_values = prcp_ds['time'].values
        _, unique_indices = np.unique(time_values, return_index=True) # Detect duplicates using NumPy
        # Sort unique indices to preserve the original order
        unique_indices = np.sort(unique_indices)
        prcp_ds = prcp_ds.isel(time=unique_indices)   # Select only the unique time entries using `isel` for efficient indexing
            
        return(prcp_ds)

        

class NoiseGenerator:
    
    def __init__(self, pcp_file, dt, pvar, wind_file, uname, vname, wsize, seed, advection,
                preprocess_pcp=None,preprocess_wind=None,verbose=False):
        self.wsize = wsize
        self.seed = seed
        self.advection = advection
        self.pcp_file = pcp_file
        self.verbose = verbose
        self.Validate = Validate(self)
        
        if preprocess_pcp!=None: # Optional precipitation post-processing step defined by user
            if verbose: print('Preprocessing precip data.')
            ds = preprocess_pcp(pcp_file)
        else:
            if len(pcp_file)==1: ds = xr.open_dataset(pcp_file).sel(time=slice(dt.strftime('%Y-%m-%d'),None))
            else: ds = xr.open_mfdataset(pcp_file).sel(time=slice(dt.strftime('%Y-%m-%d'),None))
        
        ds = self.Validate.checkPrecipField(ds,pvar) # Validate that format of precip data is correct
        self.pcp = ds
        self.pvar = pvar

        if advection:
            self.uname = uname
            self.vname = vname
            self.wind_file = wind_file
            if preprocess_wind!=None: # apply preprocessing to wind file
                if verbose: print('Preprocessing wind data.')
                dsw = preprocess_wind(wind_file)
            else:
                if len(wind_file)==1: ds = xr.open_dataset(wind_file)
                else: dsw = xr.open_mfdataset(wind_file)
            self.dsw = self.Validate.checkWindField(dsw,ds)
            

    def get_alpha(self,data0, data1):
        '''
        This function calculates the lag-1 temporal autocorrelation between two fields: data0 & data1
        '''
        lp, _ = pearsonr(data1[data1 + data0 != 0.0].flatten(), data0[data1 + data0 != 0.0].flatten())
        return lp
        ## -------------------------------------------------------------------------

    
    def _tukey(self, R, alpha):
        '''
        This function ...
        Author: Yuan Liu
        '''
        W = np.ones_like(R)
        N = min(R.shape[0], R.shape[1])
        mask1 = R < int(N / 2)
        mask2 = R > int(N / 2) * (1.0 - alpha)
        mask = np.logical_and(mask1, mask2)
        W[mask] = 0.5 * (1.0 + np.cos(np.pi * (R[mask] / (alpha * 0.5 * N) - 1.0 / alpha + 1.0)))
        mask = R >= int(N / 2)
        W[mask] = 0.0
        return W
        ## -------------------------------------------------------------------------

    
    def tukey_window_generation(self, m, n, alpha=0.2):
        '''
        This function ...
        Author: Yuan Liu
        '''
        X, Y = np.meshgrid(np.arange(n), np.arange(m))
        R = np.sqrt((X - int(n / 2)) ** 2 + (Y - int(m / 2)) ** 2)
        window_mask = self._tukey(R, alpha)
        window_mask += 1e-6  # add small value to avoid zero
        
        return window_mask
        ## -------------------------------------------------------------------------

    
    def compute_amplitude_spectrum(self, field):
        '''
        This function computes the amplitude of the power spectrum of a field.
        Author: Yuan Liu
        '''
        F = self.do_fft2(field)
        F.imag = (F.imag - np.mean(F.imag)) / np.std(F.imag)
        F.real = (F.real - np.mean(F.real)) / np.std(F.real)
        F_abs = np.abs(F)
        
        return F_abs
    
    ## -------------------------------------------------------------------------

    def do_fft2(self, array):
        return(np.fft.fft2(array))

    
    def do_ifft2(self, array):
        return(np.fft.ifft2(array))
    
    ## -------------------------------------------------------------------------

    def ffst_based_noise_generation(self, field, noise, **kwargs):
        '''
        This function replicates the power spectrum, both globally and locally, 
        of a given precipitation field in a field of white noise.
        Author: Yuan Liu
        '''
        
        win_size = self.wsize
        overlap = kwargs.get('overlap', 0.5)
        ssft_war_thr = kwargs.get('ssft_war_thr', 0.1)
        
        dim_x = field.shape[1]
        dim_y = field.shape[0]
        num_windows_y = int(np.ceil(float(dim_y) / win_size[0]))
        num_windows_x = int(np.ceil(float(dim_x) / win_size[1]))
        global_F = self.compute_amplitude_spectrum(field)
        noise_F = self.do_fft2(noise)
        
        global_noise_array = self.do_ifft2(noise_F * global_F).real
        final_noise_array = np.zeros(global_noise_array.shape)
        final_weight_array = np.zeros(global_noise_array.shape)
        
        for i in range(num_windows_y):
            
            for j in range(num_windows_x):
                
                idxi = np.zeros(2).astype(int)
                idxj = np.zeros(2).astype(int)
                idxi[0] = int(np.max((i * win_size[0] - overlap * win_size[0], 0)))
                idxi[1] = int(np.min((idxi[0] + win_size[0] + overlap * win_size[0], dim_y)))
                idxj[0] = int(np.max((j * win_size[1] - overlap * win_size[1], 0)))
                idxj[1] = int(np.min((idxj[0] + win_size[1] + overlap * win_size[1], dim_x)))
                
                window_rainfall_array = field[idxi[0]: idxi[1], idxj[0]: idxj[1]]
                curr_window_dimension = (idxi[1] - idxi[0], idxj[1] - idxj[0])
                tukey_window = self.tukey_window_generation(m=curr_window_dimension[0], n=curr_window_dimension[1])
                weighted_window_rainfall_array = window_rainfall_array * tukey_window
                wet_area_ratio = np.sum(weighted_window_rainfall_array > 0.01) / (curr_window_dimension[0] * curr_window_dimension[1])
                full_mask = np.zeros((dim_y, dim_x))
                full_mask[idxi[0]: idxi[1], idxj[0]: idxj[1]] = tukey_window
                
                if wet_area_ratio > ssft_war_thr:
                    full_masked_rainfall_array = field * full_mask
                    local_F = self.compute_amplitude_spectrum(full_masked_rainfall_array)
                    local_noise_array = self.do_ifft2(noise_F * local_F).real
                    final_noise_array += local_noise_array * full_mask
                    final_weight_array += full_mask
                
                else:
                    final_noise_array += global_noise_array * full_mask
                    final_weight_array += full_mask
        
        final_noise_array[final_weight_array > 0] /= final_weight_array[final_weight_array > 0]
        final_noise_array = (final_noise_array - np.mean(final_noise_array)) / np.std(final_noise_array)
        
        return(final_noise_array)
        
    ## -------------------------------------------------------------------------

    def getAdvectionGridData(self,xsize,ysize,ds):
        '''
        This function retrieves grids that define the distance between grid cells in meters,
        which is used in advection scheme.
        Author: Daniel Wright, Sam Hartke
        '''

        yind = np.repeat(np.arange(0, ysize), xsize).reshape(ysize, xsize).astype('int16')
        xind = np.tile(np.arange(0, xsize), ysize).reshape(ysize, xsize).astype('int16')

        # ---  Get median lat and lon points, whether these are 2D or 1D variables
        if ds['lat'].dims == ('y', 'x'):
            dlon = np.median(ds['lon'][:][0, 1:] - ds['lon'][:][0, 0:-1])
            dlat = np.median(ds['lat'][:][1:, 0] - ds['lat'][:][0:-1, 0])
            gridx = ds['lon'].values
            gridy = ds['lat'].values
        else:
            dlon = np.median(ds['lon'][1:] - ds['lon'][:-1])
            dlat = np.median(ds['lat'][1:] - ds['lat'][:-1])
            gridx,gridy=np.meshgrid(ds['lon'].values,ds['lat'].values) 
        

        # ----  Create arrays, dx and dy, with distance between grid cells in units of meters
        dx = Utils.latlon_distance(gridy.flatten(), gridx.flatten(), gridy.flatten(), gridx.flatten() + dlon)
        dx = dx.reshape(ysize, xsize)
        dy = Utils.latlon_distance(gridy.flatten(), gridx.flatten(), gridy.flatten() + dlat, gridx.flatten())
        dy = dy.reshape(ysize, xsize)
        
        return(dx, dy, xind, yind)
        
    ## -------------------------------------------------------------------------
    
    def advect_field(self, ds, ts, dt, s, d, rn, alpha, dx, dy, xind, yind, xsize, ysize):
        '''
        This function advects a noise field based on wind data and returns the advected field
        Author: Sam Hartke
        '''
        u = self.dsw[self.uname].sel(time=dt)
        deltax = u * ts
        dix = deltax // dx

        v = self.dsw[self.vname].sel(time=dt)
        deltay = -v * ts
        diy = deltay // dy

        ybefore = np.array((yind - diy) % ysize, dtype='int16')
        xbefore = np.array((xind - dix) % xsize, dtype='int16')

        new_s = (alpha * s[d - 1, ybefore, xbefore] + np.sqrt(1.0 - alpha ** 2) * rn).astype('float32')
        
        return(new_s)

    ## -------------------------------------------------------------------------

    def get_corr_noise(self, n, dt, seednum, runtag, ts = None, verbose=False,
                       wetThresh=0.05, preprocess=None, overlap_ratio = 0.3, ssft_war_thr = 0.1):
        '''
        This function creates a 3D field of uniform noise that is correlated in space and time
        based on the correlation of an input precipitation field.
        Authors: Sam Hartke, Daniel Wright, Yuan Liu
        '''
        
        # Subset precip dataset to date and timesteps selected for simulation
        dt_start = dt.strftime('%Y-%m-%d')
        end_index = self.pcp.get_index('time').get_loc(dt_start).start + n # get index of end date for simulation
        
        if n>self.pcp.time.shape[0]:
            print('Precip dataset has %d timesteps and user has requested %d for simulation.'%(self.pcp.time.shape[0],n))
            dt_end = str(self.pcp.time.values[-1])
        else:
            dt_end = str(self.pcp.time.values[end_index-1])
        
        ds = self.pcp.sel(time=slice(dt_start, dt_end))
        pcp = ds[self.pvar][:n]
        
        if self.advection:
            self.dsw[self.uname] = self.dsw[self.uname].sel(time=slice(dt_start, dt_end))[:n] # also subset wind field to these timesteps
            self.dsw[self.vname] = self.dsw[self.vname].sel(time=slice(dt_start, dt_end))[:n]
        
        if ds['lat'].dims == ('y', 'x'): xsize = ds.dims['x']; ysize = ds.dims['y']
        else: xsize = ds.dims['lon']; ysize = ds.dims['lat']
        
        field_size = xsize*ysize
        s = np.empty((n, ysize, xsize), dtype=np.float32)
        
        if self.advection: # If advection is turned on, retrieve information for advection grid
            dx, dy, xind, yind = self.getAdvectionGridData(xsize,ysize,ds)

            
        pcp0 = pcp.sel(time=dt.strftime('%Y-%m-%d'))[0, :, :].values
        
        if len(pcp0[pcp0 != 0.0]) > wetThresh*field_size:
            if verbose: print('Replicating power spectrum of first field')
            s[0, :, :] = self.ffst_based_noise_generation(field = pcp0,
                                             noise = np.random.normal(size=(ysize, xsize)))
            first_rain = True
        
        else:
            if verbose: print('Using white noise for first field')
            s[0, :, :] = np.random.normal(size=(ysize, xsize))
            first_rain = False

        # Start with an initial alpha, or temporal correlation value, of 0.9. This will be recalculated at each timestep.
        alpha = 0.9
        
        # Calculate timestep for advection [seconds] if not provided
        if (ts==None) & self.advection:
            time_index = pd.to_datetime(ds['time'].values) # Convert time variable to pandas DatetimeIndex
            time1 = time_index[0]
            time2 = time_index[1]
            time_diff = time2 - time1      # Compute the difference between the two times
            ts = time_diff.total_seconds()   # Convert the time difference to seconds

        if verbose: print('Running Semi-Lagrangian Scheme starting at %d'%decade)
        decade = dt.year

        for d in range(1, n):
        
            nextday = ds.time[d].values
            if verbose and (n%500==0): print(nextday) # Print update every 500 time steps if verbose
            dt1 = str(nextday)
            pcph = pcp[d].values
            
            if len(pcph[pcph != 0.0]) > wetThresh*field_size: # Check that this field meets the rain area threshold
                rn = self.ffst_based_noise_generation(field = pcph,
                                             noise = np.random.normal(size=(ysize,xsize)))
                alpha = self.get_alpha(pcph, pcp[d-1].values)
                first_rain = True
                recent_rain = np.copy(pcph)
                
            else:
                if verbose: print('Non-rainy day')
                # If non-rainy day, use correlation structure of last rainy day, recent_rain
                if first_rain == True:
                    rn = self.ffst_based_noise_generation(field = recent_rain,
                                             noise = np.random.normal(size=(ysize, xsize)))
                else:
                    if verbose: print('Using white noise for timestep d')
                    rn = np.random.normal(size=(ysize, xsize))

            if self.advection:
                s[d, :, :] = self.advect_field(ds, ts, dt1, s, d, rn, alpha, dx, dy, xind, yind, xsize, ysize)
            else:
                s[d, :, :] = (alpha * s[d - 1, :, :] + np.sqrt(1.0 - alpha ** 2) * rn).astype('float32')

        s = 0.5*(1+sp.special.erf((s/math.sqrt(2)))) # convert to uniform noise
        
        # Convert to xarray dataset with lat, lon values and variable 'noise'
        final_ds = pcp.to_dataset().copy()
        final_ds['noise'] = (('time','lat','lon'),s)
        
        return(final_ds.drop(self.pvar))
        
        ## -------------------------------------------------------------------------


