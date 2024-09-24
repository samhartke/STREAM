"""
Code to plot the output of STREAM simulations
"""

from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr
import scipy.stats as sp
import numpy as np

def label_plot(ax, proj, top=False,bottom=True,left=True,right=False):
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.7)
    ax.add_feature(cfeature.OCEAN, color="white", edgecolor="black")
    ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.7)

    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.25, color='grey', alpha=1., linestyle='--')
    gl.top_labels = top
    gl.bottom_labels = bottom
    gl.right_labels = right
    gl.left_labels = left


def plot_map(da, ax=None, proj=None, yaxis="lat", xaxis="lon", cmap=None, vmax=None, vmin=None, colorbar=True, top=False,bottom=True,left=True,right=False,
             colorbar_label="Precipitation [mm]", xlim=None, ylim=None):
    
    if proj is None: proj=ccrs.PlateCarree()
    if ax is None: ax = plt.axes(projection=proj)

    if colorbar == True: qm = da.plot.pcolormesh(xaxis, yaxis, cmap=cmap, ax=ax, transform=proj, vmax=vmax, vmin=vmin, add_colorbar=True,cbar_kwargs={'shrink':0.6})
    else: qm = da.plot.pcolormesh(xaxis, yaxis, cmap=cmap, ax=ax, transform=proj, vmax=vmax, vmin=vmin, add_colorbar=False)
    label_plot(ax, proj, top=top, bottom=bottom, left=left, right=right)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if colorbar:
        qm.colorbar.set_label(colorbar_label)

# ------------------------------------------------------------------------------

dt = datetime(2050, 4, 4)  # Define date to start plotting at

# ---- Define the fields that you want to plot:

## Original precipitation input field
prcp = xr.open_dataset('../example_data/gard_out_pcp.nc')['pcp'].sel(time=slice(dt.strftime('%Y-%m-%d'),None))

## Correlated noise output field
noise = xr.open_dataset('../example_output/noise_output_GARD.nc')['noise'].sel(time=slice(dt.strftime('%Y-%m-%d'),None))

# # Simulated precipitation output field
# Leaving this commented out because no precipitation was simulated in GARD example
# simPrcp = xr.open_dataset('example_precip_output.nc')['sim_prcp'].sel(time=slice(dt.strftime('%Y-%m-%d'),None))


# ---- Plot these fields for six timesteps
fig = plt.figure(figsize=(20,8))
proj = ccrs.PlateCarree()

for i in range(6):

    if 'prcp' in globals():
        ax1 = fig.add_subplot(3,6,i+1,projection=proj)
        plot_map(prcp[i,:,:],ax1,cmap='turbo',vmin=0.,vmax=15.,colorbar=False,left=False,bottom=False)
        if i==0:
            ax1.text(-0.08, 0.55, 'Input precip\nfield', va='bottom', ha='center',rotation='vertical',
                 fontsize=20,rotation_mode='anchor',transform=ax1.transAxes,fontweight='bold')

    if 'noise' in globals():
        ax1 = fig.add_subplot(3,6,i+7,projection=proj)
        plot_map(noise[i,:,:],ax1,cmap='gray',vmin=0.,vmax=1.,colorbar=False,left=False,bottom=False)
        if i==0:
            ax1.text(-0.08, 0.55, 'Output noise\nfield', va='bottom', ha='center',rotation='vertical',
                 fontsize=20,rotation_mode='anchor',transform=ax1.transAxes,fontweight='bold')

    if 'simPrcp' in globals():
        ax1 = fig.add_subplot(3,6,i+13,projection=proj)
        plot_map(simPrcp[i,:,:],ax1,cmap='turbo',vmin=0.,vmax=5.,left=False,bottom=False)
        if i==0:
            ax1.text(-0.08, 0.55, 'Output precip\nfield', va='bottom', ha='center',rotation='vertical',
                 fontsize=20,rotation_mode='anchor',transform=ax1.transAxes,fontweight='bold')

plt.subplots_adjust(hspace=0.25)
fig.savefig('STREAM_output_GARD.jpg',bbox_inches='tight',dpi=1200)
plt.show()


