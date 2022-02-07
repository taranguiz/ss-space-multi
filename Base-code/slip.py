#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 13:24:45 2018

functions to go with strike-slip model

dependencies: numpy, matplotlib

author: nadine g reitman
email: nadine.reitman@colorado.edu
"""
###-------------------------------------------------------------------------###
#%% import modules
###-------------------------------------------------------------------------###

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] =(1,1,1,1)  # white plot backgrounds
plt.rcParams['figure.facecolor']=(1,1,1,0) # clear figure backgrounds

###-------------------------------------------------------------------------###
#%% define functions
###-------------------------------------------------------------------------###

def calcOFD(fault_zone_width,max_slip,fault_loc,y_star,dxy,ymax,nrows,plot=False,save_plot=False):
    y = np.linspace(0,ymax,nrows) # zone to plot on y/grid axis
    b = 0. # center on this point on x/slip axis
    a = max_slip/2. # this divides slip between both sides of the fault.
    y_zero = fault_loc # center on this point on y/grid axis
    y_star = y_star # length scale over which OFD decay occurs
    slip = b + a*np.tanh((y-y_zero)/y_star) # calculate ofd profile with hyperbolic tangent

    if plot:
        plt.plot(slip,y,color='navy')
        plt.axhline(fault_loc,0,1,color='k',linestyle='--',linewidth=1,)
        plt.grid(color='lightgray',linestyle='--')
        plt.ylabel('distance (m)')
        plt.xlabel('slip (m)')
        plt.title('Strike-Slip Displacement Profile')
        plt.tight_layout()
        if save_plot:
            plt.savefig('ofd_profile.png', dpi=500,transparent=True)
        plt.show()

    return slip

def calcSlipRegime (tmax, dt, interval, total_slip=0, slip_rate=0, slip_type='characteristic',std=1,seed=None,plot=False):
    '''function to make an array (size = tmax/dt) of slip values based on different earthquake recurrence models.
    slip_type can be 'characteristic', 'random_time', 'random_slip', 'random_random', or 'creep'.
    defaults to characteristic.
    WARNING: random options are not physics-based (i.e., do not depend on previous earthquake time or size)
        characteristic = constant time and slip  - same time interval between earthquakes of equal magnitude until reach total_slip
        creep = characteristic with slip every dt
        random_time: random time and predictable slip - random time interval between earthquakes of equal magnitude - sums to total_slip
        random_slip: predictable time and random slip - same time interval between earthquakes of random magnitude - sums to total_slip
        random_random: random time intervals and random slip amount - sums to total_slip
        random_std: random sampling of normal distributions with centers at interval and std=std. interval and std should be in same time units.
    interval = slip interval for characteristic and random_slip in years, use 1 for creep and <= 0 for random time
    tmax = total time in years
    total_slip = total slip in meters. calculated based on tmax and slip rate if not entered.
    slip_rate = slip speed in meters/year. calculated from tmax and total_slip if not entered.
    either total_slip or slip_rate is required.
    '''
    # seed random generator
    if seed:
        np.random.seed(seed)

    # first make sure there are numbers entered
    if total_slip==slip_rate==0:
        raise Warning('slip is zero. must enter value for total_slip or slip_rate')

    # calculate slip_speed based on inputs
    if slip_rate>0:
        slip_rate = slip_rate
        print('slip rate is '+str(np.round(slip_rate,5))+' m/yr')
    else:
        slip_rate = np.float(total_slip) / tmax
        print('slip rate is '+str(np.round(slip_rate,5))+' m/yr')


    # calculate total_slip based on inputs
    if total_slip>0:
        total_slip = total_slip
        print('total slip is '+str(total_slip)+' meters')
    else:
        total_slip = np.float(tmax * slip_rate)
        print('total slip is '+str(total_slip)+' meters')

    # checking that rates make sense
    if slip_rate*tmax != total_slip:
        raise Warning("entered values for slip_rate and total_slip don't match!")

    # initialize time & slip arrays
    timesteps = np.int(tmax/dt)
    slip = np.zeros(timesteps, dtype=float)

    if slip_type=='creep' or interval == 1:
        print('creeping')
        slip_interval = 1        # slip every this many years, 1 for creep
        slip_freq = 1            # slip interval in timesteps
        number_events = np.int(timesteps/slip_freq)
        slip_per_event = np.float(total_slip)/number_events
        slip[0::slip_freq] = slip_per_event

    elif slip_type=='characteristic':
        print('characteristic slip')
        slip_interval = interval                        # slip every this many years
        slip_freq = np.int(slip_interval/dt)            # slip interval in timesteps
        number_events = np.int(timesteps/slip_freq)
        slip_per_event = np.float(total_slip)/number_events
        slip[1:-1:slip_freq] = slip_per_event
        if np.sum(slip) > total_slip:
            slip[1] = 0

    elif slip_type=='random_time':
        # this does random event spacing, but magnitude of slip is constant & is not based on time since last event
        print('random time')
        prob_slip = np.random.random_sample(timesteps) #
        prob_slip[prob_slip<=0.95] = 0
        number_events = np.count_nonzero(prob_slip)
        slip_per_event =  np.float(total_slip)/number_events
        slip[prob_slip>0] = 1
        slip[:] *= slip_per_event

    elif slip_type=='random_slip':
        # this does random slip_per_event based on given interval, sum to total_slip, but not AT ALL based on time since or size of previous eq
        print('random slip')
        slip_interval = interval                        # slip every this many years
        slip_freq = np.int(slip_interval/dt)            # slip interval in timesteps
        number_events = np.int(timesteps/slip_freq)
        random_events = np.random.random(number_events)
        random_events *= total_slip/np.sum(random_events)
        slip[1::slip_freq] = random_events
        slip_per_event = random_events

    elif slip_type=='random_random':
        # random timing and random amount slip, sums to total_slip, not AT ALL based on time since or size of previous event
        print('random time & slip')
        prob_slip = np.random.random_sample(timesteps) #
        prob_slip[prob_slip<=0.95] = 0
        number_events = np.count_nonzero(prob_slip)
        random_events = np.random.random(number_events)
        random_events *= total_slip/np.sum(random_events)
        slip[prob_slip>0] = random_events
        slip_per_event = random_events

    elif slip_type=='random_std':
        print('slip based on random sampling of many normal distributions with std=std and centers at interval. slip per event is constant.')
        slip_interval = interval
        number_events = np.int(tmax/slip_interval) # calc number of earthquakes
        slip_per_event = np.float(total_slip)/number_events # calc slip_per_event based on

        # define locations (in time) for normal distribution centers
        loc = slip_interval * np.arange(0,number_events,1)
        # keep slip_interval in model_time, not timesteps, so that std numbers are more intuitive (i.e., std 100 = 100 years, not 100/dt years)

        # get normal distriubtion from numpy.random.normal
        out = np.random.normal(loc,std)
        out = out/dt # clean up out - put in timesteps b/c slip is in timesteps
        out[out<0] *= -1 # clean up out - make any negative values positive

        # apply out to slip_regime
        for v in out:
            if np.int(v) < timesteps:
                slip[np.int(v)] = slip_per_event

    else: print('WARNING: CHECK SPELLING OF SLIP_TYPE')

    slip_slipped = np.sum(slip)
    eqs_happened = np.sum(slip>0)

    if slip_slipped < total_slip:
        print('WARNING: SLIP IS LESS THAN DESIRED')

    print('total slip slipped is %s meters in %s earthquakes' %(slip_slipped,eqs_happened))

    if plot:
        plt.plot(range(timesteps), slip)
        plt.xlabel('time (ka)')
        plt.ylabel('slip (m)')
        plt.show()

    return slip, slip_per_event


def calcCOV(slip_regime, dt):
    '''Returns COV.
    Function to calculate COV from a slip_regime array of the type used in strike_slip_model.py
    The grid slips (an earthquake!) when slip_regime>0 = intervals[value>0]. the index of intervals lets us get at time
    between earthquakes (recurrence).'''

    # get list of [index,value] for slip_regime
    intervals = list(map(list, enumerate(slip_regime)))

    # swap index and value for each item in intervals
    for item in intervals:
        item[0], item[1] = item[1], item[0]

    # make intervals an array
    intervals = np.asarray(intervals,dtype=float)

    # keep only the earthquakes (intervals[value>0])
    earthquakes = intervals[:][intervals[:,0]>0]

    # keep only interval indexes. get rid of intervals[value] --> we don't need value now that we have intervals in timesteps
    intervals = earthquakes[:,1]

    # initiate recurrence array
    recurrence = np.zeros(len(intervals)-1)

    # calulate recurrence interval (in timesteps) by differencing next and current intervals for all of interval
    for i in range(len(recurrence)):
        recurrence[i] = intervals[i+1] - intervals[i]

    # convert recurrence in timesteps to years
    recurrence = recurrence * dt

    # calculate cov
    cov = np.round(np.std(recurrence) / np.mean(recurrence),3)

    return cov
