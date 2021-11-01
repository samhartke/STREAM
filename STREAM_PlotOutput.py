# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 08:42:54 2020

@author: Sam

Script to:
    - plot fields in simulated precipitation ensemble
    - create movie from simulated precipitation ensemble

"""

import os
from netCDF4 import Dataset,num2date
from datetime import date, timedelta
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import rc,cm
rc('text', usetex=True)



#==============================================================================

def plot_background(ax):
    
    ax.set_extent([-100., -85., 35., 45.])
    
    #Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

    ax.add_feature(states_provinces, edgecolor='gray')
    
    return ax

#==============================================================================
#
# USE THIS SECTION TO INDICATE WHAT TIME PERIOD AND WHAT FORMAT TO
# PLOT SIMULATED PRECIPITATION OUTPUT FOR
#
#------------------------------------------------------------------------------


# indicate whether to create video of results or plot of results
video = True  # this creates a video showing precipitation fields over time
plots = False  # this creates a plot of several timesteps of precipitation fields

# indicate whether to save plotted output, this must be True for video output
save = True

# start date to create plots or video for
dt_start = date(2013,6,8)
h_start = 0 # hour to start at, this is usually 0 for a video

# number of timesteps to move forward in video or plot
ts = 2*24 # usually this is <10 for plots and some multiple of 24 for a video

# ---  input file names  ---

wd = "C:/path/to/STREAMcode/"

refFname = wd + "NexradIV2013_0.1deg.hourly.nc"  # path for reference precipitation data

obsFname = wd + "IMERG2013.hourly.nc"   # path for observed precipitation data


noiseFname = wd + "noise_20130601_20130614.nc"  # path for noise ensemble

STREAMFname = wd + "STREAM_20130601_20130614.nc"   # path for STREAM precipitation ensemble



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#==============================================================================

dt_end = dt_start + timedelta(hours=ts-1) # end date of simulation to plot/create video for



# ----  retrieve ensemble data ----

ds = Dataset(STREAMFname)

file_start = num2date(ds.variables['time'][0],ds.variables['time'].units)
# getting indices of hourly data that we want to select from specific date range
dt_i = 24*(dt_start - date(file_start.year,file_start.month,file_start.day)).days + h_start 


nens = len(ds.variables['prcp'][:,0,0,0])

# select two ensemble members to plot/create movies for
n = (1,3)

# select data over desired time period
simPrcp = ds.variables['prcp'][n,dt_i:(dt_i + ts),:,:]
ds=None


# ----  retrieve simulated noise  -----
ds = Dataset(noiseFname)
file_start = num2date(ds.variables['time'][0],ds.variables['time'].units)
# getting indices of hourly data that we want to select from specific date range
dt_i = 24*(dt_start - date(file_start.year,file_start.month,file_start.day)).days + h_start 

q = ds.variables['q'][n,dt_i:(dt_i + ts),:,:]
ds=None


# ----  retrieve IMERG and Stage IV data ----

# open IMERG file and get data over Iowa City and Cedar Rapids for date range
ds = Dataset(obsFname)
file_start = num2date(ds.variables['time'][0],ds.variables['time'].units)
# getting indices of hourly data that we want to select from specific date range
dt_i = 24*(dt_start - date(file_start.year,file_start.month,file_start.day)).days + h_start 

imerg = ds.variables['prcp'][:,:,dt_i:(dt_i + ts)]
imerg[imerg<0.1] = 0.
ds=None

# open Stage IV file and get data over Iowa City and Cedar Rapids for date range
ds = Dataset(refFname)
file_start = num2date(ds.variables['time'][0],ds.variables['time'].units)
# getting indices of hourly data that we want to select from specific date range
dt_i = 24*(dt_start - date(file_start.year,file_start.month,file_start.day)).days + h_start 

stIV = ds.variables['prcp'][:,:,dt_i:(dt_i + ts)]
stIV[stIV<0.1] = 0.
ds=None

#%%

if plots == True:
    
    if ts > 8:
        print('Too many timesteps to plot')
    
    else:
        
        fig, axarr = plt.subplots(nrows=ts, ncols=6, figsize=(17, 11), constrained_layout=True,
                              subplot_kw={'projection': ccrs.PlateCarree()})
        img_extent = (-100.,-85.,35.,45.)
        axlist = axarr.flatten()
        for ax in axlist:
            plot_background(ax)
        
        for t in range(ts):
            axlist[6*t].set_title('IMERG %d:00'%((h_start+t)%24),fontsize=16)
            axlist[6*t].imshow(np.flip(imerg[:,:,t],axis=0), cmap='Blues',vmin=0,vmax=10, extent=img_extent, transform=ccrs.PlateCarree())
            rmse = np.sqrt(np.nanmean((stIV[:,:,t]-imerg[:,:,t])**2))
            axlist[6*t].annotate('RMSE %.01f'%rmse, xy=(0.45,0.02),xycoords = 'axes fraction',fontsize=15,color="r")
            axlist[6*t].axis('off')
            
            axlist[6*t+1].set_title('Stage IV %d:00'%((h_start+t)%24),fontsize=16)
            axlist[6*t+1].imshow(np.flip(stIV[:,:,t],axis=0), cmap='Blues',vmin=0,vmax=10, extent=img_extent, transform=ccrs.PlateCarree())
            axlist[6*t+1].axis('off')
            

            axlist[6*t+2].set_title('Simulated Noise %d:00'%((h_start+t)%24),fontsize=16)
            axlist[6*t+2].imshow(np.flip(q[0,t,:,:],axis=0), cmap=cm.gray,vmin=0,vmax=1., extent=img_extent, transform=ccrs.PlateCarree())
            axlist[6*t+2].axis('off')
            
            axlist[6*t+3].set_title('Simulated Noise %d:00'%((h_start+t)%24),fontsize=16)
            axlist[6*t+3].imshow(np.flip(q[1,t,:,:],axis=0), cmap=cm.gray,vmin=0,vmax=1., extent=img_extent, transform=ccrs.PlateCarree())
            axlist[6*t+3].axis('off')

            
            axlist[6*t+4].set_title('Simulated Prcp %d:00'%((h_start+t)%24),fontsize=16)
            axlist[6*t+4].imshow(np.flip(simPrcp[0,t,:,:],axis=0), cmap='Blues',vmin=0,vmax=10, extent=img_extent, transform=ccrs.PlateCarree())
            rmse = np.sqrt(np.nanmean((stIV[:,:,t]-simPrcp[0,t,:,:])**2))
            axlist[6*t+4].annotate('RMSE %.01f'%rmse, xy=(0.45,0.02),xycoords = 'axes fraction',fontsize=15,color="r")
            axlist[6*t+4].axis('off')
            
            axlist[6*t+5].set_title('Simulated Prcp %d:00'%((h_start+t)%24),fontsize=16)
            axlist[6*t+5].imshow(np.flip(simPrcp[1,t,:,:],axis=0), cmap='Blues',vmin=0,vmax=10, extent=img_extent, transform=ccrs.PlateCarree())
            rmse = np.sqrt(np.nanmean((stIV[:,:,t]-simPrcp[1,t,:,:])**2))
            axlist[6*t+5].annotate('RMSE %.01f'%rmse, xy=(0.45,0.02),xycoords = 'axes fraction',fontsize=15,color="r")
            axlist[6*t+5].axis('off')
        
        if save==True:
            
            if not os.path.exists(wd+'figures'):
                os.makedirs(wd+'figures')
            
            fig.savefig(wd + 'figures/SimulatedFields_%s_%dhrs.pdf'%(dt_start.strftime('%Y%m%d'),ts))
            fig.savefig(wd + 'figures/SimulatedFields_%s_%dhrs.png'%(dt_start.strftime('%Y%m%d'),ts), transparent=True, dpi=1200)
            
        plt.show()


if video == True:
    
    #wd = "C:/Users/Sam/OneDrive/QuantileWork/"
    
    # set paths to folders with specific dates - create folders if necessary
    vidpath = wd + "videoImages/simResults/simResults%s_%s/"%(dt_start.strftime('%Y%m%d'),dt_end.strftime('%Y%m%d'))
    
    if not os.path.exists(vidpath):
        os.makedirs(vidpath)
    
    for t in range(ts):
        
        dt = dt_start + timedelta(hours=t)
        
        fig, axarr = plt.subplots(nrows=3, ncols=2, figsize=(12, 15),
                          subplot_kw={'projection': ccrs.PlateCarree()})
        
        axlist = axarr.flatten()
        
        img_extent = (-100.,-85.,35.,45.)
        
        for ax in axlist:
            plot_background(ax)
        
        axlist[0].set_title('IMERG %s %d:00'%(dt.strftime('%Y-%m-%d'),t%24),fontsize=16)
        axlist[0].imshow(np.flip(imerg[:,:,t],axis=0), cmap='Blues',vmin=0,vmax=15, extent=img_extent, transform=ccrs.PlateCarree())
        rmse = np.sqrt(np.nanmean((stIV[:,:,t]-imerg[:,:,t])**2))
        axlist[0].annotate('%.01f'%rmse, xy=(0.8,0.02),xycoords = 'axes fraction',fontsize=17)
        axlist[0].axis('off')
        
        axlist[1].set_title('Stage IV %s %d:00'%(dt.strftime('%Y-%m-%d'),t%24),fontsize=16)
        axlist[1].imshow(np.flip(stIV[:,:,t],axis=0), cmap='Blues',vmin=0,vmax=15, extent=img_extent, transform=ccrs.PlateCarree())
        axlist[1].axis('off')
        
        axlist[2].set_title('STREAM %s %d:00'%(dt.strftime('%Y-%m-%d'),t%24),fontsize=16)
        axlist[2].imshow(np.flip(simPrcp[0,t,:,:],axis=0), cmap='Blues',vmin=0,vmax=15, extent=img_extent, transform=ccrs.PlateCarree())
        rmse = np.sqrt(np.nanmean((stIV[:,:,t]-simPrcp[0,t,:,:])**2))
        axlist[2].annotate('%.01f'%rmse, xy=(0.8,0.02),xycoords = 'axes fraction',fontsize=17)
        axlist[2].axis('off')
        
        axlist[3].set_title('STREAM %s %d:00'%(dt.strftime('%Y-%m-%d'),t%24),fontsize=16)
        axlist[3].imshow(np.flip(simPrcp[1,t,:,:],axis=0), cmap='Blues',vmin=0,vmax=15, extent=img_extent, transform=ccrs.PlateCarree())
        rmse = np.sqrt(np.nanmean((stIV[:,:,t]-simPrcp[1,t,:,:])**2))
        axlist[3].annotate('%.01f'%rmse, xy=(0.8,0.02),xycoords = 'axes fraction',fontsize=17)
        axlist[3].axis('off')
        
        axlist[4].set_title('Simulated Noise %s %d:00'%(dt.strftime('%Y-%m-%d'),t%24),fontsize=16)
        axlist[4].imshow(np.flip(q[0,t,:,:],axis=0), cmap=cm.gray,vmin=0,vmax=1., extent=img_extent, transform=ccrs.PlateCarree())
        axlist[4].axis('off')
        
        axlist[5].set_title('Simulated Noise %s %d:00'%(dt.strftime('%Y-%m-%d'),t%24),fontsize=16)
        axlist[5].imshow(np.flip(q[1,t,:,:],axis=0), cmap=cm.gray,vmin=0,vmax=1., extent=img_extent, transform=ccrs.PlateCarree())
        axlist[5].axis('off')
        
        
        fig.savefig(vidpath + 'simResults_%d%02d%02d_%02d.png'%(dt.year, dt.month, dt.day, t%24))
        plt.close(fig)
    
    t = int(ts*0.5) # length of movie - alloting 12 sec per day
    
    # run images together into a movie
    cmd = "python tk-img2video.py -o videoImages/simResults%s_%s.mp4 -d videoImages/simResults/simResults%s_%s -t %d"%(dt_start.strftime('%Y%m%d'),dt_end.strftime('%Y%m%d'),dt_start.strftime('%Y%m%d'),dt_end.strftime('%Y%m%d'),t)
    os.system(cmd)
    
    # delete images folder
    
    print('Video simResults%s_%s.mp4 has been saved.'%(dt_start.strftime('%Y%m%d'),dt_end.strftime('%Y%m%d')))
    




