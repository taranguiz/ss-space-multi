#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 2019
author: nadine g reitman
email: nadine.reitman@colorado.edu

Landscape evolution model with lateral displacement.

written in python 3.

dependencies: numpy, yaml, matplotlib, landlab, pickle, slip function included in this archive

"""

###-------------------------------------------------------------------------###
#%% import modules
###-------------------------------------------------------------------------###

import time, os                       # track run time, chdirs, read args from command line
import numpy as np                         # math with numpy
import yaml                                # for reading in a parameter file
import matplotlib.pyplot as plt            # plot with matplotlib
plt.rcParams['axes.facecolor'] =(1,1,1,1)  # white plot backgrounds
plt.rcParams['figure.facecolor']=(1,1,1,0) # clear figure backgrounds
plt.rcParams['xtick.top']=True # plot ticks on top
plt.rcParams['ytick.right']=True # plot ticks on right side
plt.rcParams['ytick.direction']='in' # tick marks face in
plt.rcParams['xtick.direction']='in' # tick marks face in

from slip import calcSlipRegime, calcOFD, calcCOV # my function for making tectonic regimes and ofd profiles

from landlab import RasterModelGrid, imshow_grid # landlab grid components
from landlab.components import LinearDiffuser, DepressionFinderAndRouter
#from landlab.components import FlowRouter, FastscapeEroder
from landlab.components import FlowAccumulator, FastscapeEroder
import pickle as pickle
from landlab.io import read_esri_ascii 

###-------------------------------------------------------------------------###
#%% load parameters from YAML parameter file
###-------------------------------------------------------------------------###

# load parameter configuration from config.yaml file
#config = yaml.load(open('config_files/config.yaml','r')) # use this to run fron config/here
config = yaml.load(open('config_files/config.yaml','r'),Loader=yaml.FullLoader) # EDITED LINE; The newer version of yaml requires Loader=yaml.FullLoader
#config = yaml.load(open(sys.argv[1],'r')) # use this line to read file from command line
#config = yaml.load(open('config.yaml','r')) # EDITED LINE; config.yaml is in the same level as strike_slip_model.py

# saving options
save_pickle = config['saving']['save_pickle'] # save final topography as pickle
save_ascii = config['saving']['save_ascii'] # save final topography as ASCII file - for re-import to landlab
model_name = config['saving']['model_name'] # model name for naming outputs - ztopo, avgZ, displacement, movie
save_location = 'model_output/%s' %model_name # location to save output

# time parameters
dt = config['time']['dt'] # timestep in years
tmax = config['time']['tmax'] # total time in years

# grid parameters
xmax = config['grid']['xmax'] # x grid dimension in meters
ymax = config['grid']['ymax'] # y grid dimension in meters
dxy = config['grid']['dxy'] # grid step in meters

# geomorph parameters
kappa = config['geomorph']['kappa'] # hillslope diffusivity; bob says this is good value for DV alluvium. see Nash, Hanks papers. m2/yr diffusivity / transport coefficient for soil diffusion
K_sp = config['geomorph']['K_sp'] # stream power for FastscapeEroder
threshold = config['geomorph']['threshold'] # threshold for FastscapeEroder

# strike-slip parameters
slip_interval = config['strike-slip']['slip_interval'] # how often an earthquake occurs [years]
total_slip = config['strike-slip']['total_slip'] # total slip for entire model time [meters]
#slip_rate = config['strike-slip']['silp_rate'] # slip rate [m/yr]
slip_type = config['strike-slip']['slip_type'] # slip type for calc slip regime function. options are characteristic, creep, random_time, random_slip, random_random
control = config['strike-slip']['control'] # make a model with no tectonics to compare how landscape acts without perturbations
uplift_rate = config['strike-slip']['uplift_rate'] # background relative rock uplift rate [m/yr]
std = config['strike-slip']['std']
seed = config['strike-slip']['seed']
slip_regime = config['strike-slip']['slip_regime']

# ofd parameters
fault_zone_width = config['ofd']['fault_zone_width'] # define width of fault zone (ofd zone) in meters. use 0.000001 for none b/c otherwise get divide by zero error. (distance from fault on either side) [is this half or total? --> total, but not actually used. only used to get ystar = fzw/5]
y_star = fault_zone_width/7. # length scale of OFD decay [meters].

# plotting parameters (grid plotting & initial conditions)
figsize = config['plotting']['figsize'] # size of grid plots
shrink = config['plotting']['shrink'] # amount of colorbar shrinkage for plots (0-1). 1 = not shrunk. 0 = nonexistent.
limits = config['plotting']['limits'] # elevation limits for grid plots
plots = config['plotting']['plots'] # how often to save a frame for the movie. 1 = every timestep.
#plot_inits = config['plotting']['plot_inits'] # make initial condition plots?


###-------------------------------------------------------------------------###
#%% INITIALIZE (set parameters and boundary conditions)
###-------------------------------------------------------------------------###

###### INITIALIZE GRID ##########################
nrows = int(ymax/dxy) # EDITED LINE
ncols = int(xmax/dxy) # EDITED LINE
# make a new topo grid - uncomment next line
#grid = RasterModelGrid((nrows,ncols),dxy) 

# load topo from previous run
(grid, z) = read_esri_ascii('ztopo_1000x500y1.asc',name='topographic__elevation') 


# set boundary conditions on grid
# east, north, west, south,  True = closed; False = open
grid.set_closed_boundaries_at_grid_edges(True, True, True, False)

####### SET UP SAVING IN THE MODEL_OUTPUT DIRECTORY ##############
# make a directory if it doesn't exist and switch to model_name directory
try:
    os.chdir(save_location)
except:
    os.mkdir('model_output/%s' %model_name)
    os.chdir(save_location)

#print('saving into: %s' %os.getcwd()) # print to screen
print('saving into: %s' %(os.getcwd()), file=open('out_%s.txt' %model_name, 'w')) # print to a file


#### print parameters to file ####
print('dt: %s' %dt, file=open('out_%s.txt' %model_name, 'a'))
print('tmax: %s' %tmax, file=open('out_%s.txt' %model_name, 'a'))
print('dxy: %s' %dxy, file=open('out_%s.txt' %model_name, 'a'))
print('kappa: %s' %kappa, file=open('out_%s.txt' %model_name, 'a'))
print('Ksp: %s' %K_sp, file=open('out_%s.txt' %model_name, 'a'))
print('threshold: %s' %threshold, file=open('out_%s.txt' %model_name, 'a'))
print('slip type: %s' %slip_type, file=open('out_%s.txt' %model_name, 'a'))
print('saving frame every %s timesteps' %plots, file=open('out_%s.txt' %model_name, 'a'))
print('',file=open('out_%s.txt' %model_name, 'a'))

###### SET MODEL TIME PARAMS ####################
model_time = np.arange(0,tmax,dt)


###### SET GEOMORPH PARAMS ######################
m = 0.5                 # exponent on area in stream power equation
n = 1.                  # exponent on slope in stream power equation

###### SET TECTONICS PARAMS #####################
# put a strike-slip fault half way up Y axis
fault_loc = int(ymax / 2.) # EDITED LINE; Remove 'np.'

# row of nodes that are main fault trace
fault_nodes = np.where(grid.node_y==fault_loc)[0]

# calculate slip regime and get max_slip = slip_per_event if slip_regime not loaded already

# make a slip regime and slip_per_event
slip_regime, slip_per_event = calcSlipRegime(tmax, dt, slip_interval,
                                             std=std, total_slip=total_slip,
                                             slip_type=slip_type, seed=seed)

print('slip per event is '+ str(np.round(slip_per_event,2))+' meters', file=open('out_%s.txt' %model_name, 'a'))

# calculate COV of slip_regime
cov = calcCOV(slip_regime,dt)
print('COV is %s' %cov, file=open('out_%s.txt' %model_name, 'a'))

# calculate cumulative_slip
cumulative_slip = np.zeros((len(model_time)),dtype=float)
for i in range(len(model_time)-1):
    if slip_regime[i] > 0:
        cumulative_slip[i] += slip_regime[i]
    cumulative_slip[i+1] = cumulative_slip[i]


max_slip = slip_per_event # [meters]

# calc ofd with function
# calculate ofd slip profile, this is length(nrows)
ofd_profile = calcOFD(fault_zone_width,max_slip,fault_loc,y_star,dxy,ymax,nrows,plot=False)

###### SET UP TO TRACK DISPLACEMENT #############
# because the grid is discretized into pixels, we need to count how much deformation has occurred over an earthquake
# and move a pixel after the accumulated deformation is larger than than the pixel length
accum_disp = np.zeros(nrows) # start with no accumulated displacement
accum_disp_total = np.zeros(shape=(nrows,ncols)) # also track total accumulated displacement
displacement = grid.add_zeros('node','accumulated__displacement') # add field to grid to track accumulated displacement

# This is an array for counting how many pixels need to be moved
nshift = np.zeros(nrows,dtype=int)

######## INITIALIZE LANDLAB COMPONENTS ##########
linear_diffuser = LinearDiffuser(grid,linear_diffusivity=kappa) # hillslope diffusion
#flow_router = FlowRouter(grid) # flow routing # Commented out
flow_router = FlowAccumulator(grid, flow_director='D8') # ADDED LINE; Replaced old FlowRouter component with FlowDirectorSteepest component
#depression_finder = DepressionFinderAndRouter(grid) # pit finding and filling...it's slow
fill = DepressionFinderAndRouter(grid)
fastscape_eroder = FastscapeEroder(grid,K_sp=K_sp, m_sp=m, n_sp=n, threshold_sp=threshold) # stream power erosion



###### PLOT INITIAL CONDITIONS ##################

# plot slip_regime and cumulative slip
fig = plt.subplots(2,1, figsize=(5,7))
ax1 = plt.subplot(2,1,1)
ax1.plot(model_time, slip_regime,color='steelblue',linewidth=1.0)
ax1.set_ylabel('slip (m)')
ax1.set_xlabel('time (yrs)')
ax1.set_title('slip regime',fontsize=10)
ax2 = plt.subplot(2,1,2)
ax2.plot(model_time,cumulative_slip,color='goldenrod')
ax2.set_ylabel('slip (m)')
ax2.set_xlabel('time (yrs)')
ax2.set_title('cumulative slip',fontsize=10)
ax2.text(0,28,'cov: %s' %cov)
plt.tight_layout()
plt.savefig('slip_%s.png' %model_name, dpi=300, facecolor=(1,1,1,0)) # white facecolor with clear figure background


# plot ofd profile
fig = plt.subplots(figsize=(5.5,5))
plt.plot(ofd_profile,np.linspace(0,ymax,nrows),color='navy',linewidth=1.2)
plt.axhline(fault_loc,0,1,color='k',linestyle='--',linewidth=0.5)
plt.grid(color='lightgray',linestyle='--')
plt.ylabel('distance (m)')
plt.xlabel('slip (m)')
plt.title('Strike-Slip Displacement Profile')
#plt.tight_layout()
plt.savefig('ofd_profile_%s.png' %model_name,dpi=300,facecolor=(1,1,1,0))

###-------------------------------------------------------------------------###
#%%  RUN
###-------------------------------------------------------------------------###

# set start time for keeping track how long a run takes
start_time = time.time()
print("--- start time: %s ---" %(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())),file=open('out_%s.txt' %model_name, 'a'))


# calculate number of iterations based on total time
iterations = len(model_time)

# save pngs to make movies later
fig = plt.figure(figsize=figsize)
imshow_grid(grid,z,cmap='terrain',limits=limits,grid_units=['m','m'],shrink=shrink)
plt.axhline(fault_loc,0,xmax,color='k',linestyle='--',linewidth = 0.5)
plt.title('Topography after 0 years')
plt.text(xmax-110,-80, 'total slip: 0 m')
plt.savefig('topo_%s_yrs_%s.png' %(0,model_name),dpi=300,facecolor=(1,1,1,0))

# starting an eq counter for the random tectonic regimes
# b/c max_slip is same length as number eq,
# so need to access the right value of max_slip (i.e., the max_slip for that eq)
eq_counter = 0

# RUN THE MODEL
for i in range(iterations):

    # if this is a time when tectonics happens, do tectonics
    if slip_regime[i] > 0:

        # Take the landlab grid elevations and reshape into a box nrows x ncols
        z_reshape = np.reshape(z[:],[nrows,ncols])

        # Calculate the offset that has accumulated for this time/event
        # this is used to cound how much to shift the grid for this time/event
        # after slip happens, the amount slipped is subtracted from accum_disp
        if len(ofd_profile) == nrows: 
            accum_disp += ofd_profile
            for j in range(ncols): accum_disp_total[:,j]+= ofd_profile

        else:
            accum_disp += ofd_profile[eq_counter,:]
            for j in range(ncols): accum_disp_total[:,j]+= ofd_profile[eq_counter,:]


        # keep track of total accumulated displacement
        accum_disp_total_reshape = np.reshape(accum_disp_total,nrows*ncols)

        # save total accumulated displacement in a field on the grid
        displacement[:] = accum_disp_total_reshape

        # count number of pixels to be moved this time/event
        nshift[:] = np.floor(accum_disp/dxy)

        # now scan up the landscape row by row looking for offset
        for r in range(nrows): # change xrange to range for Py3

            # check if the accumulated offset for a row is larger than a pixel
            if accum_disp[r] >= dxy or accum_disp[r] <= -dxy:

                # move the row over by the number of pixels of accumulated offset
                z_reshape[r,:] = np.roll(z_reshape[r,:],nshift[r])

                # subtract the offset pixels from the displacement
                accum_disp[r] -= dxy * nshift[r]

        # Reshape the elevation box into an array for landlab
        z_new = np.reshape(z_reshape, nrows*ncols)

        # feed new z back into landlab
        z[:] = z_new

        # move the eq_counter ahead
        eq_counter += 1

    # now do the landscape evolution stuff
    # diffuse landscape via soil diffusion
    linear_diffuser.run_one_step(dt)
    # calculate flow routes across new landscape
    flow_router.run_one_step()
    # erode landscape based on routed flow and stream incision
    fastscape_eroder.run_one_step(dt)
    # uplift by background uplift rate
    z[:] += uplift_rate * dt
    # make sure bottom row stays at 0 elevation
    z[grid.node_y==0] = 0

    # save topo for every 'plots' timestep
    if i % plots == 0:
        print('iteration '+str(i),file=open('out_%s.txt' %model_name, 'a'))
    	# save topo to ascii
        grid.save('topo_%s_yrs_%s.asc' %(int((i*dt)+dt),model_name), names=['topographic__elevation'])
        # calculate total displacement at one grid cell with the max displacement
        slip_amount = np.round(-displacement[1]*2,1)
        # plot figure of this timestep
        fig = plt.figure(figsize=figsize)
        imshow_grid(grid,z,cmap='terrain',limits=limits,grid_units=['m','m'],shrink=shrink) # eventually update this with hillshade background? gdal not installed in same env right now
        plt.title('Topography after '+str(int((i*dt)+dt))+' years')
        plt.text(xmax-110,-80, 'total slip: '+str(slip_amount)+' m')
        plt.savefig('topo_%s_yrs_%s.png' %(int((i*dt)+dt),model_name),dpi=300,facecolor=(1,1,1,0))


# save image for last frame of movie
fig = plt.figure(figsize=figsize)
imshow_grid(grid,z,cmap='terrain',limits=limits,grid_units=['m','m'],shrink=shrink)
plt.title('Topography after '+str(int((i*dt)+dt))+' years')
plt.text(xmax-110,-80, 'total slip: '+str(np.sum(slip_regime))+' m')
plt.savefig('topo_%s_yrs_%s.png' %(int((i*dt)+dt),model_name), dpi=300,facecolor=(1,1,1,0)) # save final topo as image


###### SAVE TOPOGRAPHY ########

# save final topography as pickle
if save_pickle:
    pickle.dump(np.array(z),open('topo_final_%s.p' %model_name,'wb'))

# save final topography as asc (ASCII)
if save_ascii:
    grid.save('topo_final_%s.asc' %model_name, names=['topographic__elevation'])


###### CLEAN UP ########
#calculate time to run model and print to output
run_time = np.round((time.time() - start_time)/60.,2)
print("--- run time: %s minutes ---" %run_time,file=open('out_%s.txt' %model_name, 'a'))
print("--- end time: %s ---" %(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())),file=open('out_%s.txt' %model_name, 'a'))

# go back to top-level directory
os.chdir('../../')