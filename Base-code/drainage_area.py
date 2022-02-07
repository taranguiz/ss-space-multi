#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:23:56 2019

postproccess stike-slip model output topography w/ landlab to get drainage area 
saves drainage area as ascii file

dependencies: numpy, matplotlib, landlab

author: nadine g reitman
email: nadine.reitman@colorado.edu
"""

######################
#%% import modules ###
######################

import yaml
import numpy as np                         # math with numpy
np.random.seed(123456)                     # seed random 
import matplotlib.pyplot as plt            # plot with matplotlib
plt.rcParams['axes.facecolor'] =(1,1,1,1)  # white plot backgrounds
plt.rcParams['figure.facecolor']=(1,1,1,0) # clear figure backgrounds
from landlab.components import FlowRouter,DepressionFinderAndRouter
import matplotlib.colors as colors
from landlab.io import read_esri_ascii 

######################
#%% set parameters ###
######################

# load parameter configuration from config.yaml file
config = yaml.load(open('config_files/config.yaml','r')) # use this to read a config file/run from here
#config = yaml.load(open(sys.argv[1],'r')) # use this line to read file from command line

model_name = config['saving']['model_name'] # model name for naming outputs
output_loc = 'model_output/%s' %model_name # location to save output

# geomorph parameters
kappa = config['geomorph']['kappa'] # hillslope diffusivity; bob says this is good value for DV alluvium. see Nash, Hanks papers. m2/yr diffusivity / transport coefficient for soil diffusion
K_sp = config['geomorph']['K_sp'] # stream power for FastscapeEroder
threshold = config['geomorph']['threshold'] # threshold for FastscapeEroder
m = 0.5
n = 1.
uplift_rate = config['strike-slip']['uplift_rate'] # background relative rock uplift rate [m/yr]

# time parameters
dt = config['time']['dt'] # timestep in years

# grid parameters
xmax = config['grid']['xmax'] # x grid dimension in meters
ymax = config['grid']['ymax'] # y grid dimension in meters
dxy = config['grid']['dxy'] # grid step in meters
nrows = np.int(ymax/dxy)
ncols = np.int(xmax/dxy)

# load topo
(grid, z) = read_esri_ascii('%s/topo_final_%s.asc' %(output_loc,model_name),name='topographic__elevation') 

# set boundary conditions on grid
# east, north, west, south
grid.set_closed_boundaries_at_grid_edges(True, True, True, False)

# initialize landlab components
fr = FlowRouter(grid, method='D8')
df = DepressionFinderAndRouter(grid,routing='D8')

############################################
#%% run one step of landscape evolution  ###
############################################
reroute_flow = True
fr.run_one_step()
df.map_depressions(pits='flow__sink_flag',reroute_flow=reroute_flow) # call depression finder like this to id pits without changing flow

# calc and plot drainage area

# DRAINAGE AREA
da = grid["node"]["drainage_area"]
da = np.reshape(da,[nrows,ncols])

# AREA ^ M
Am = da**m

# plot both drainage area and A^m
fig = plt.subplots(2,1,figsize=(10,8))
plt.subplot(2,1,1)
plt.imshow(da,cmap='inferno',origin='lower',
           alpha=1,norm=colors.Normalize(vmax=25000),
           interpolation='none')
plt.colorbar(shrink=.5,label=r'$m^2$')
plt.ylabel('y (m)')
plt.title('Drainage Area')

plt.subplot(2,1,2)
plt.imshow(Am,cmap='inferno',origin='lower',alpha=1,interpolation='none')
plt.colorbar(shrink=.5)
plt.title(r'A$^m$')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()

# save figure
plt.savefig('%s/da_%s.png' %(output_loc,model_name),dpi=600,facecolor=(1,1,1,0))
plt.show()

# save drainage area as ascii
grid.save('%s/da_%s.asc' %(output_loc,model_name), names=['drainage_area'])
