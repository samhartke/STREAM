# The Space-Time Rainfall Error and Autocorrelation Model (STREAM)

# Introduction
The Space-Time Rainfall Error and Autocorrelation Model (STREAM) combines space-time correlation structures of satellite precipitation fields with a pixel scale precipitation error model to generate precipitation ensembles that can “bracket” the magnitude and replicate the correlation structure of ground reference (i.e. true) rainfall. STREAM was developed and evaluated for a study area in the central U.S. using the NASA satellite precipitation product, IMERG-Early, at an hourly 0.1º scale. For more information about STREAM methodology, see Hartke et al. 2022. 

Hartke, S. H., Wright, D. B., Li, Z., Maggioni, V., Kirschbaum, D. B., & Khan, S. (2022). Ensemble representation of satellite precipitation uncertainty using a nonstationary, anisotropic autocorrelation model. Water Resources Research, 58, e2021WR031650. https://doi.org/10.1029/2021WR031650 

# Layout
There are two files required to simulate spatiotemporally correlated noise fields using a given precipitation field and associated wind fields:
- STREAM_Main.py
- STREAM_NoiseGeneration.py

Optional files to generate and visualize simulated precipitation from correlated noise fields:
- STREAM_PlotOutput.py
- STREAM_PrecipSimulation.py
- CSGD_TrainErrorModel.py


### Retrieving data for STREAM 
- IMERG Early data downloaded from GESDISC – variable ‘precipitationCal’ 
- MERRA2 data downloaded from GESDISC – variables ‘U850’ and ‘V850’ in m/s 
 
### Required python packages for STREAM 
Required packages: pysteps, netCDF4, tqdm

Installing pysteps python library 
- For anaconda users: https://anaconda.org/conda-forge/pysteps 
- Otherwise: https://pysteps.readthedocs.io/en/v1.0.0/user_guide/install_pysteps.html 
 
### Input parameters and filenames for STREAM 
**nEns** – number of ensemble members to generate 

**dt** – date to begin STREAM simulation at 

**ts** – timesteps to run STREAM simulation for 
 
**wd** – directory containing input data for STREAM 

**obsInFname** – name of netcdf file containing satellite precipitation data 

**windInFname** – name of netcdf file containing wind speed data in u- and v- directions 

**paramsInFname** – name of netcdf file containing CSGD error model parameters 
 
**noiseOutFname** – name of netcdf file to save noise ensemble to 

**precipOutFname** – name of netcdf file to save STREAM precipitation ensemble to
 
### Running STREAM 
Fill in all parameters and filenames in the Input Parameters Section of STREAM_main.py 

Run STREAM_main.py 
 
### Formatting input data for STREAM 
**obsInFname** should be a netcdf file with variable ‘prcp’ with dimensions (time,y,x)

**windInFname** should be a netcdf file with variables ‘uWind’ and ‘vWind’ with dimensions (time,y,x) at the same resolution as obsInFname 

**paramsInFname** should be a netcdf file with dimensions (y,x) at the same resolution as obsInFname and variables 'clim1','clim2','clim3', 'par1','par2','par3','par4','par5' 
 
### Optional: Training CSGD error model for STREAM 
Use functions in CSGD_utilities.py to train CSGD error model. 

An example training script is provided in CSGD_trainErrorModel.py.

For more details on the pixel-scale CSGD error model, see: 

Wright, D. B., Kirschbaum, D. B., & Yatheendradas, S. (2017). Satellite Precipitation Characterization, Error Modeling, and Error Correction Using Censored Shifted Gamma Distributions. Journal of Hydrometeorology, 18(10), 2801–2815. https://doi.org/10.1175/JHM-D-17-0060.1 

# License
STREAM is distributed under the MIT License:

Copyright 2022 Hartke

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to use without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

The software is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, arising from, out of or in connection with the Software or the use or other dealings in the Software.

# Support disclaimer
This code is made available on an as-is basis. There is no guarantee that it will run on a given platform or if it does, that it will run correctly.
