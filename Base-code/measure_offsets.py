#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:58:03 2019

function to measure offsets from strike-slip model output topography

dependencies: numpy, matplotlib, rasterio, scipy, scikit-image, slip function included in this archive

author: nadine g reitman
email: nadine.reitman@colorado.edu

"""
#%% import modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from skimage import measure
import pickle
import rasterio as rio
#import yaml

import warnings
warnings.filterwarnings('ignore')


# figure settings
plt.rcParams['axes.facecolor'] =(1,1,1,1)  # white plot backgrounds
plt.rcParams['figure.facecolor']=(1,1,1,0) # clear figure backgrounds
plt.rcParams['xtick.top']=True # plot ticks on top
plt.rcParams['ytick.right']=True # plot ticks on right side
plt.rcParams['ytick.direction']='in' # tick marks face in
plt.rcParams['xtick.direction']='in' # tick marks face in
plt.rcParams["errorbar.capsize"] = 3

#%% define functions

def measure_offsets(da_ascii,fzw,dxy,real_max_slip,name,save_loc,years=10000,
                    channel_cutoff=1000,clean=False,left=False,plot=True,):
    ''' 
    function to measure offsets given an ascii file of drainage area. 
    saves output as pngs, thalwags.p (thalwag indices), offsets.p (offset msrmts & locations)
    
    INPUTS:
    da_ascii = filename ending in .asc (ESRI ASCII format)
    fzw = fault zone width 
    dxy = grid pixel size
    real_max_slip = imposed total slip in model to evaluate how well auto method does
    name = string to name output data and figures
    years = years of simulation 
    channel_cutoff = value of drainage area above which is a channel
    clean = whether or not to recalculate offset stats and pdf with outlier removed.
    left = is this a left-lateral fault? default=False for: no, it's right-lateral
    plot=True or False for plot & save figures or not
    save_loc = directory to save output in
    '''
    ###########################################    
    ### open drainage area and get channels ###
    ###########################################
    da = rio.open(da_ascii)
    da = da.read(1)
    
    ymax = np.shape(da)[0]
    fault_loc = np.int(ymax/2)
    
    # channels are da>1000
    channels = np.copy(da)
    channels[channels<channel_cutoff] = np.nan
    channels[channels>=channel_cutoff] = 1000
    
    #############################################
    ### calc connected components on channels ###
    #############################################  
    conncomps = measure.label(channels, background=0, connectivity=2)

    
    conncomps_above = conncomps[fault_loc-(fzw),:]
    conncomps_below = conncomps[fault_loc+fzw,:]  


        
    #############################################
    ### extract subset from above/below fault ###
    #############################################
    
    ### NOTICE SWITCH IN ABOVE BELOW +- B/C WEIRD INDEXING
    subset_above = channels[fault_loc-(fzw),:]
    subset_below = channels[fault_loc+fzw,:]

    # do binary thalwags arrays for yes/no thalwag/not
    thals_above = np.zeros_like(subset_above)
    thals_below = np.zeros_like(subset_below)
    
    for i,v in enumerate(subset_above):
        if subset_above[i]>=1000: thals_above[i] = 1
        
    for i,v in enumerate(subset_below):
        if subset_below[i]>=1000: thals_below[i] = 1
        
    # initiate offset array
    offsets = np.zeros_like(thals_above)
    
    # get a list of index,value for thals_below and above, then compare
    intervals = list(map(list, enumerate(thals_below)))  # get list of [index,value]
    for item in intervals:
        item[0], item[1] = item[1], item[0]  # swap index with value
    intervals = np.asarray(sorted(intervals),dtype=float) # sort by thalwag value low to high
    events = intervals[:][intervals[:,0]>0] # find indexs of thalwags/events
    thal_index_below = events[:,1] # get rid of events, keep only index column
    
    intervals = list(map(list, enumerate(thals_above)))  # get list of [index,value]
    for item in intervals:
        item[0], item[1] = item[1], item[0]  # swap index with value
    intervals = np.asarray(sorted(intervals),dtype=float) # sort by thalwag value low to high
    events = intervals[:][intervals[:,0]>0] # find indexs of thalwags/events
    thal_index_above = events[:,1] # get rid of events, keep only index column
    
    ### clean up thal_index to get rid of connected indices ###
    # could update this using connected components now
    # thal_index_above
    for i,v in enumerate(thal_index_above[:-1]):
        if v+1 == thal_index_above[i+1]:
            thal_index_above[i+1] = np.nan
    for i,v in enumerate(thal_index_above[:-2]):
        if v+2 == thal_index_above[i+2]:
            thal_index_above[i+2] = np.nan
    for i,v in enumerate(thal_index_above[:-3]):
        if v+3 == thal_index_above[i+3]:
            thal_index_above[i+3] = np.nan
    for i,v in enumerate(thal_index_above[:-4]):
        if v+4 == thal_index_above[i+4]:
            thal_index_above[i+4] = np.nan
    for i,v in enumerate(thal_index_above[:-5]):
        if v+5 == thal_index_above[i+5]:
            thal_index_above[i+5] = np.nan
    # thal_index_below
    for i,v in enumerate(thal_index_below[:-1]):
        if v+1 == thal_index_below[i+1]:
            thal_index_below[i+1] = np.nan
    for i,v in enumerate(thal_index_below[:-2]):
        if v+2 == thal_index_below[i+2]:
            thal_index_below[i+2] = np.nan
    for i,v in enumerate(thal_index_below[:-3]):
        if v+3 == thal_index_below[i+3]:
            thal_index_below[i+3] = np.nan
    for i,v in enumerate(thal_index_below[:-4]):
        if v+4 == thal_index_below[i+4]:
            thal_index_below[i+5] = np.nan
    for i,v in enumerate(thal_index_below[:-5]):
        if v+5 == thal_index_below[i+5]:
            thal_index_below[i+5] = np.nan
            
    # remove any nan entries       
    thal_index_above = thal_index_above[thal_index_above>0]
    thal_index_below = thal_index_below[thal_index_below>0]
    
    # remove any thalwegs in the below array that are part of the same channel system and close by
    thal_index_below = np.flip(thal_index_below) # flip to reverse order
    for i,v in enumerate(thal_index_below[:-1]):
        if conncomps_below[np.int(thal_index_below[i])] == conncomps_below[np.int(thal_index_below[i+1])]:
            if thal_index_below[i] <= thal_index_below[i+1]+10:
                thal_index_below[i] = 0
    
    thal_index_below = thal_index_below[thal_index_below>0]
    thal_index_below = np.flip(thal_index_below)
    
    # save thalweg data
    pickle.dump([thal_index_above,thal_index_below],open('%s/thalwags_%s_%s.p' %(save_loc,name,years),'wb'))
    
    # make offsets length of above array (implicit assumption that it is longer)
    offsets = np.zeros_like(thal_index_above)
        
    # plot conncomps and thal_indexes
    if plot:
        
        y_above = np.zeros_like(thal_index_above)
        y_below = np.zeros_like(thal_index_below)
        y_above[:] = fault_loc-fzw
        y_below[:] = fault_loc+fzw
        
        plt.figure(figsize=(12,8))
        plt.imshow(conncomps,cmap='nipy_spectral_r')
        plt.axhline(fault_loc+fzw,0,1000,color='k',linestyle='--',linewidth=1)
        plt.axhline(fault_loc-fzw,0,1000,color='k',linestyle='--',linewidth=1)
        plt.plot(thal_index_above,y_above,'k*',markersize=5)
        plt.plot(thal_index_below,y_below,'k*',markersize=5)
        plt.colorbar(shrink=0.62)
        plt.title('channel connected components')
        plt.savefig('%s/connected_channels_%s_%s.png' %(save_loc,name,years),dpi=600,transparent=True)
        #plt.show()
        plt.clf() 
        
    ##################################        
    ### calc offsets from channels ###
    ##################################
    
    # initialize array to store offset locs
    offset_locs = np.zeros(shape=(len(offsets),2))
    
    # set up iterator for below array
    j = 0 
       
    # iterate through the above/longer dataset 
    
    # THIS IS FOR LEFT LATERAL:
    if left:
        print('left-lateral implementation needs to be updated')
    
    # THIS IS FOR RIGHT LATERAL:   
    else:
        # iterate through the above-fault thalwag index array
        for i,v in enumerate(thal_index_above): 
            
            # if above is right of below AND above is same connected channel system
            if (thal_index_above[i] > thal_index_below[j]) & (conncomps_above[np.int(v)] == conncomps_below[np.int(thal_index_below[j])]): 
                
                # record an offset measurement and it's location
                # offset is above index - below index (corrected for dx later)
                offsets[i] = thal_index_above[i] - thal_index_below[j] 
                # offset location is above index, below index
                offset_locs[i,0],offset_locs[i,1] = thal_index_above[i], thal_index_below[j]
                
                # then move the below iterator forward
                # check that we won't exceed the length of the below dataset when move iterator forward
                if (j+1) < len(thal_index_below): 
                    # move iterator forward for below dataset 
                    j+=1 
                # don't move below iterator forward at end of below array
                else: j=j 
             
            # else if above is right of below AND above is NOT same connected channel system    
            elif (thal_index_above[i] > thal_index_below[j]) & (conncomps_above[np.int(v)] != conncomps_below[np.int(thal_index_below[j])]):
                # don't record an offset
                # move below iterator forward b/c this is a beheaded/orphaned channel
                # check that we won't exceed the length of the below dataset when move iterator forward
                if (j+1) < len(thal_index_below):
                    # move iterator forward for below dataset 
                    j=j+1 
                                        
                    # check that we satisfy requirements for recording an offset with new J:
                    if (thal_index_above[i] > thal_index_below[j]) & (conncomps_above[np.int(v)] == conncomps_below[np.int(thal_index_below[j])]):
                        
                        # record an offset measurement for i and just updated j and it's location
                        # offset is above index - below index (corrected for dx later)
                        offsets[i] = thal_index_above[i] - thal_index_below[j] 
                        # offset location is above index, below index
                        offset_locs[i,0],offset_locs[i,1] = thal_index_above[i], thal_index_below[j]
                       
                    # if we don't satisfy reqiurements for an offset with new J, move J up again and try again:
                    elif (thal_index_above[i] > thal_index_below[j]) & (conncomps_above[np.int(v)] != conncomps_below[np.int(thal_index_below[j])]):
                        j=j+1
                        
                        # check satisfy requirements for offset with 2nd new J
                        if (thal_index_above[i] > thal_index_below[j]) & (conncomps_above[np.int(v)] == conncomps_below[np.int(thal_index_below[j])]):
                        
                            # record an offset measurement for i and just updated j and it's location
                            # offset is above index - below index (corrected for dx later)
                            offsets[i] = thal_index_above[i] - thal_index_below[j] 
                            # offset location is above index, below index
                            offset_locs[i,0],offset_locs[i,1] = thal_index_above[i], thal_index_below[j]
                    
                        
                # don't move below iterator forward at end of below array
                else: j=j 
            
            # else if above is left of below AND it's part of the same connected channel system as the previous below thalwag, then it's a valid offest (stream capture)                  
            elif (thal_index_above[i] < thal_index_below[j]) & (conncomps_above[np.int(v)] == conncomps_below[np.int(thal_index_below[j-1])]): 
                
                # record an offset here with previous below index (this is a stream capture)
                # SOMEHOW FLAG THIS OFFSET AS A CAPTURE, MAYBE DON"T INCLUDE IN SOME ANALYSIS B/C LIKELY TO BE LARGE!
                offsets[i] = thal_index_above[i] - thal_index_below[j-1]
                # offset location is above index, previous below index
                offset_locs[i,0],offset_locs[i,1] = thal_index_above[i], thal_index_below[j-1]
                # don't move below iterator forward b/c it's already on the channel system ahead(right) of the current above thalwag
                j=j 
                
            # else if above is left of below AND it's NOT part of the same connected channel system as previous, it's an apparent left offset
            elif (thal_index_above[i] < thal_index_below[j]) & (conncomps_above[np.int(v)] != conncomps_below[np.int(thal_index_below[j-1])]):
                # record a negative offset to signify left lateral? then need to figure out how to deal with that in processing later on
                # don't move the below iterator forward b/c this is apparent left offset
                j=j
            
            print(offsets[i],offset_locs[i]) # print for debugging
    
    
    # remove zeros/nans
    # this captures and removes negative numbers
    offset_locs = offset_locs[offsets>0] # this turns it into 1 long array
    offset_locs = offset_locs.reshape(np.int(np.sum(offset_locs>0)/2),2) # reshape back to 2-column array
    offsets = offsets[offsets>0]
    
    # scale offsets by grid size
    offsets = offsets * dxy
    
    ########################   
    ### save offset data ###
    ########################
    pickle.dump([offsets,offset_locs],open('%s/offsets_%s_%s.p' %(save_loc,name,years),'wb'))
    
    ################################   
    ### plot channels w/ offsets ###
    ################################    
    
    if plot:
        y_above = np.zeros_like(offsets)
        y_below = np.zeros_like(offsets)
        y_above[:] = fault_loc-fzw
        y_below[:] = fault_loc+fzw
        
        fig = plt.subplots(figsize=(12,8))
        plt.imshow(da,cmap='Greys',vmin=0,vmax=1000)
        plt.imshow(channels,cmap='winter') # rainbow gives purple, raibow_r gives red
        plt.axhline(fault_loc+fzw,0,1000,color='r',linestyle='--')
        plt.axhline(fault_loc-fzw,0,1000,color='r',linestyle='--')
        plt.scatter(offset_locs[:,0],y_above)#c=offsets)
        plt.scatter(offset_locs[:,1],y_below)#c=offsets)
        plt.savefig('%s/channels_%s_%s.png' %(save_loc,name,years),dpi=600,facecolor=(1,1,1,0))
        plt.show()
        plt.clf()

    #######################    
    ### calculate stats ###
    #######################
    
    # calculate mean
    mean = np.round(np.mean(offsets),2)
    #print('mean: %s' %mean)
    
    # calculate rmse
    real_offsets = np.zeros_like(offsets)
    real_offsets[:]=real_max_slip
    rmse = np.round(np.sqrt(mean_squared_error(real_offsets, offsets)),2)
    #print('RMSE: %s' %rmse)
    
    # calculate standard deviation
    std = np.round(np.std(offsets),2)
    #print('StDev: %s' %std)
    
    # make PDFs 
    offsets_sorted = np.sort(offsets)
    pdf = norm.pdf(offsets_sorted, mean, std)
    
    #######################    
    ### clean offsets?? ###
    #######################  
    if clean:
        # throw out outliers = any >or< than 2*std
        offsets_clean = np.copy(offsets)
        offsets_clean[offsets>(mean+2*std)] = np.nan
        offsets_clean[offsets_clean<(mean-2*std)] = np.nan
        
        # also save locations of the clean set of offsets --> ugh do later
        offset_locs_clean = np.copy(offset_locs)
        offset_locs_clean = offset_locs_clean[offsets_clean>0]
        
        # remove nan entries       
        offsets_clean = offsets_clean[offsets_clean>0]
        
        # then recalculate stats
        # calculate mean
        mean_clean = np.round(np.mean(offsets_clean),2)
        #print('mean: %s' %mean)
        
        # calculate rmse
        real_offsets_clean = np.zeros_like(offsets_clean)
        real_offsets_clean[:]=real_max_slip    
        rmse_clean = np.round(np.sqrt(mean_squared_error(real_offsets_clean, offsets_clean)),2)
        #print('RMSE: %s' %rmse)
        
        # calculate standard deviation
        std_clean = np.round(np.std(offsets_clean),2)
        #print('StDev: %s' %std)
        
        # make PDFs 
        offsets_sorted_clean = np.sort(offsets_clean)
        pdf_clean = norm.pdf(offsets_sorted_clean, mean_clean, std_clean)
        
   
    ########################   
    ### plot offset data ###
    ########################
    if plot:
        
        ########################### 
        ### displacement vs fzw ###
        ########################### 
        x = np.zeros_like(offsets)
        x[:] = fzw
        
        fig,ax = plt.subplots(figsize=(8,6))
        plt.axhline(real_offsets[0],color='indianred',linestyle='--')
        
        plt.errorbar(x,offsets,yerr=std,lw=1,marker='o',linestyle=None,color='royalblue')#mfc='k', mec='black',ecolor='k' )#ms=4, mew=2)
        plt.plot(pdf+fzw, offsets_sorted,'-',color='royalblue')
        plt.fill_betweenx(offsets_sorted,fzw,pdf+fzw,alpha=.3,color='royalblue')
        plt.text(0.02,.95,'mean: %s'%mean,transform=ax.transAxes,color='royalblue')
        plt.text(0.02,.90,'RMSE: %s'%rmse,transform=ax.transAxes,color='royalblue')
        plt.text(0.02,.85,'std: %s'%std,transform=ax.transAxes,color='royalblue')
        
        if clean:
            x_clean = np.zeros_like(offsets_clean)
            x_clean[:] = fzw
            plt.errorbar(x_clean,offsets_clean,yerr=std_clean,lw=1,marker='o',linestyle=None,color='k')#mfc='k', mec='black',ecolor='k' )#ms=4, mew=2)
            plt.plot(pdf_clean+fzw, offsets_sorted_clean,ls='--',color='k')
            plt.fill_betweenx(offsets_sorted_clean,fzw,pdf_clean+fzw,linestyle='--',alpha=.3,color='k')
            plt.text(0.2,.95,'mean clean: %s'%mean_clean,transform=ax.transAxes)
            plt.text(0.2,.90,'RMSE clean: %s'%rmse_clean,transform=ax.transAxes)
            plt.text(0.2,.85,'std clean: %s'%std_clean,transform=ax.transAxes)
        
        
        plt.text(fzw-.45,real_max_slip+1,'actual offset',color='indianred')
        plt.xlim(fzw-.5,fzw+.5)
        plt.ylabel('apparent offset measurement (m)')
        plt.xlabel('fault zone width (m)')
        plt.savefig('%s/offsets_%s_%s.png'%(save_loc,name,years),dpi=450,facecolor=(1,1,1,0))
        #plt.show()
        plt.clf()
        
        #################################
        ### displacement along strike ###
        #################################
        # x location of each offset is mean of above and below thalwag indices
        x = np.zeros_like(offsets)
        for i in range(len(offset_locs)):
            x[i] = (offset_locs[i,0]+offset_locs[i,1]) / 2.
        
        fig,ax = plt.subplots(figsize=(6,4))
        plt.fill_between((0,1000),mean+std,mean-std,alpha=0.1,color='k')      
        plt.axhline(real_offsets[0],color='indianred',linestyle='--',linewidth=1,label='real')
        plt.axhline(mean,color='k',linestyle='-.',linewidth=1,label='mean')        
        plt.plot(x,offsets,'.') 
        
        if clean: 
            # this part isn't working all the way
    #        x_clean = np.zeros_like(offsets_clean)
    #        for i in range(len(offsets_clean)):
    #            x_clean[i] = (offset_locs_clean[i,0]+offset_locs_clean[i,1]) / 2.
    #        plt.fill_between(x,mean_clean+std_clean,mean_clean-std_clean,alpha=0.3,color='royalblue')
            plt.axhline(mean_clean,linestyle='-',linewidth=1,label='clean')
    
 
        plt.xticks((0,100,200,300,400,500,600,700,800,900,1000))
        plt.yticks((0,5,10,15,20,25,30,35,40,45,50,55,60))
        plt.xlabel('distance along strike (m)')
        plt.ylabel('apparent offset (m)')
        #plt.grid(which='both',color='lightgray',linewidth=.5,linestyle='--')
        plt.xlim(0,1000)
        plt.ylim(-3,60)
        plt.legend(loc='upper right',fontsize=10)
        plt.savefig('%s/offsets_along_strike_%s_%s.png' %(save_loc,name,years),dpi=300,facecolor=(1,1,1,0))
        plt.show()
    
        

    return offsets, offset_locs, thal_index_above, thal_index_below

#%% def for small offsets
def measure_small_offsets(da_ascii,fzw,dxy,real_max_slip,name,save_loc,years=10000,channel_cutoff=1000,clean=False,left=False,plot=True):
    ''' 
    function to measure offsets given an ascii file of drainage area. 
    saves output as pngs, thalwags.p (thalwag indices), offsets.p (offset msrmts & locations)
    
    INPUTS:
    da_ascii = filename ending in .asc (ESCRI ASCII format)
    fzw = fault zone width 
    dxy = grid pixel size
    real_max_slip = imposed total slip in model to evaluate how well auto method does
    name = string to name output data and figures
    years = years of simulation 
    channel_cutoff = value of drainage area above which is a channel
    clean = whether or not to recalculate offset stats and pdf with outlier removed. not working right yet?
    left = is this a left-lateral fault? default=False for: no, it's right-lateral
    plot=True or False for plot & save figures or not
    save_loc = directory to save output in
    '''
    ###########################################    
    ### open drainage area and get channels ###
    ###########################################
    da = rio.open(da_ascii)
    da = da.read(1)
    
    ymax = np.shape(da)[0]
    fault_loc = np.int(ymax/2)
    
    # channels are da>1000
    channels = np.copy(da)
    channels[channels<channel_cutoff] = np.nan
    channels[channels>=channel_cutoff] = 1000
    
    #############################################
    ### calc connected components on channels ###
    #############################################  
    conncomps = measure.label(channels, background=0, connectivity=2)
    
    conncomps_above = conncomps[fault_loc-(fzw),:]
    conncomps_below = conncomps[fault_loc+fzw,:]

        
    #############################################
    ### extract subset from above/below fault ###
    #############################################
    
    ### NOTICE SWITCH IN ABOVE BELOW +- B/C WEIRD INDEXING
    subset_above = channels[fault_loc-(fzw),:]
    subset_below = channels[fault_loc+fzw,:]

    # do binary thalwags arrays for yes/no thalwag/not
    thals_above = np.zeros_like(subset_above)
    thals_below = np.zeros_like(subset_below)
    
    for i,v in enumerate(subset_above):
        if subset_above[i]>=1000: thals_above[i] = 1
        
    for i,v in enumerate(subset_below):
        if subset_below[i]>=1000: thals_below[i] = 1
        
    # initiate offset array
    offsets = np.zeros_like(thals_above)
    
    # get a list of index,value for thals_below and above, then compare
    intervals = list(map(list, enumerate(thals_below)))  # get list of [index,value]
    for item in intervals:
        item[0], item[1] = item[1], item[0]  # swap index with value
    intervals = np.asarray(sorted(intervals),dtype=float) # sort by thalwag value low to high
    events = intervals[:][intervals[:,0]>0] # find indexs of thalwags/events
    thal_index_below = events[:,1] # get rid of events, keep only index column
    
    intervals = list(map(list, enumerate(thals_above)))  # get list of [index,value]
    for item in intervals:
        item[0], item[1] = item[1], item[0]  # swap index with value
    intervals = np.asarray(sorted(intervals),dtype=float) # sort by thalwag value low to high
    events = intervals[:][intervals[:,0]>0] # find indexs of thalwags/events
    thal_index_above = events[:,1] # get rid of events, keep only index column
    
    ### clean up thal_index to get rid of connected indices ###
    # could update this using connected components now
    # thal_index_above
    for i,v in enumerate(thal_index_above[:-1]):
        if v+1 == thal_index_above[i+1]:
            thal_index_above[i+1] = np.nan
    for i,v in enumerate(thal_index_above[:-2]):
        if v+2 == thal_index_above[i+2]:
            thal_index_above[i+2] = np.nan
    for i,v in enumerate(thal_index_above[:-3]):
        if v+3 == thal_index_above[i+3]:
            thal_index_above[i+3] = np.nan
    for i,v in enumerate(thal_index_above[:-4]):
        if v+4 == thal_index_above[i+4]:
            thal_index_above[i+4] = np.nan
    for i,v in enumerate(thal_index_above[:-5]):
        if v+5 == thal_index_above[i+5]:
            thal_index_above[i+5] = np.nan
    # thal_index_below
    for i,v in enumerate(thal_index_below[:-1]):
        if v+1 == thal_index_below[i+1]:
            thal_index_below[i+1] = np.nan
    for i,v in enumerate(thal_index_below[:-2]):
        if v+2 == thal_index_below[i+2]:
            thal_index_below[i+2] = np.nan
    for i,v in enumerate(thal_index_below[:-3]):
        if v+3 == thal_index_below[i+3]:
            thal_index_below[i+3] = np.nan
    for i,v in enumerate(thal_index_below[:-4]):
        if v+4 == thal_index_below[i+4]:
            thal_index_below[i+5] = np.nan
    for i,v in enumerate(thal_index_below[:-5]):
        if v+5 == thal_index_below[i+5]:
            thal_index_below[i+5] = np.nan
            
    # remove any nan entries       
    thal_index_above = thal_index_above[thal_index_above>0]
    thal_index_below = thal_index_below[thal_index_below>0]
    
    # remove any thals in the below array that are part of the same channel system and close by
    thal_index_below = np.flip(thal_index_below) # flip to reverse order
    for i,v in enumerate(thal_index_below[:-1]):
        if conncomps_below[np.int(thal_index_below[i])] == conncomps_below[np.int(thal_index_below[i+1])]:
            if thal_index_below[i] <= thal_index_below[i+1]+10:
                thal_index_below[i] = 0
    
    thal_index_below = thal_index_below[thal_index_below>0]
    thal_index_below = np.flip(thal_index_below)
    
    # save thalwag data
    pickle.dump([thal_index_above,thal_index_below],open('%s/thalwags_%s_%s.p' %(save_loc,name,years),'wb'))
    
    # make offsets length of above array (implicit assumption that it is longer)
    offsets = np.zeros_like(thal_index_above)
        
    # plot conncomps and thal_indexes
    if plot:
        
        y_above = np.zeros_like(thal_index_above)
        y_below = np.zeros_like(thal_index_below)
        y_above[:] = fault_loc-fzw
        y_below[:] = fault_loc+fzw
        
        plt.figure(figsize=(12,8))
        plt.imshow(conncomps,cmap='nipy_spectral_r')
        plt.axhline(fault_loc+fzw,0,1000,color='k',linestyle='--',linewidth=1)
        plt.axhline(fault_loc-fzw,0,1000,color='k',linestyle='--',linewidth=1)
        plt.plot(thal_index_above,y_above,'k*',markersize=5)
        plt.plot(thal_index_below,y_below,'k*',markersize=5)
        plt.colorbar(shrink=0.62)
        plt.title('channel connected components')
        plt.savefig('%s/connected_channels_%s_%s.png' %(save_loc,name,years),dpi=600,transparent=True)
        #plt.show()
        plt.clf() 
        
    ##################################        
    ### calc offsets from channels ###
    ##################################
    
    # initialize array to store offset locs
    offset_locs = np.zeros(shape=(len(offsets),2))
    
    # set up iterator for below array
    j = 0 
       
    # iterate through the above/longer dataset 
    
    # iterate through the above-fault thalwag index array
    for i,v in enumerate(thal_index_above): 
        
#       print(i,j) # print for debugging
        
       # if above and below are same connected channel system
       if conncomps_above[np.int(v)] == conncomps_below[np.int(thal_index_below[j])]:
           
#           print(i,j) # print for debug
            
           # record an offset measurement and it's location
           # offset is above index - below index (corrected for dx and right vs left lateral later)
           offsets[i] = thal_index_above[i] - thal_index_below[j] 
           # offset location is above index, below index
           offset_locs[i,0],offset_locs[i,1] = thal_index_above[i], thal_index_below[j]
            
           # then move the below iterator forward
           # check that we won't exceed the length of the below dataset when move iterator forward
           if (j+1) < len(thal_index_below): 
               # move iterator forward for below dataset 
               j=j+1                
           # don't move below iterator forward at end of below array
           else: j=j
         
       # if above and below are NOT same connected channel system
       elif conncomps_above[np.int(v)] != conncomps_below[np.int(thal_index_below[j])]:
           
           # check if above is same channel system as j-1 below:
           if (conncomps_above[np.int(v)] == conncomps_below[np.int(thal_index_below[j-1])]):
               # if so, record an offset
#               print(i,j-1) # print for debug
               offsets[i] = thal_index_above[i] - thal_index_below[j-1]
               offset_locs[i,0],offset_locs[i,1] = thal_index_above[i], thal_index_below[j-1]
               j=j
           
           # if above is not same channel system as j-1,   
           elif conncomps_above[np.int(v)] != conncomps_below[np.int(thal_index_below[j-1])] :
               # don't record an offset
               # move below iterator forward b/c it's beheaded channel  
               # check that we won't exceed the length of the below dataset when move iterator forward
               if (j+1) < len(thal_index_below): 
                   j=j+1
                   
                   if conncomps_above[np.int(v)] == conncomps_below[np.int(thal_index_below[j])]:
#                       print(i,j) # print for debug
                       offsets[i] = thal_index_above[i] - thal_index_below[j]
                       offset_locs[i,0],offset_locs[i,1] = thal_index_above[i], thal_index_below[j]
                       
                   elif conncomps_above[np.int(v)] != conncomps_below[np.int(thal_index_below[j])]:
                       if (j+1) < len(thal_index_below): 
                           j=j+1
#                           print(i,j) # print for debug
                           offsets[i] = thal_index_above[i] - thal_index_below[j]
                           offset_locs[i,0],offset_locs[i,1] = thal_index_above[i], thal_index_below[j]

               # don't move iterator forward at end of below dataset
               else: j=j
               
       print(offsets[i],offset_locs[i]) # print for debugging
    
    
    # remove zeros/nans and implement left/right for positive/negative    
    
    # THIS IS FOR LEFT LATERAL:
    if left:
        if slip == 0:
            offset_locs = offset_locs[offsets<=0] # this turns it into 1 long array
            offset_locs = offset_locs.reshape(np.int(np.sum(offset_locs>0)/2),2) # reshape back to 2-column array
            offsets = offsets[offsets<=0]
        else:
            offset_locs = offset_locs[offsets<0] # this turns it into 1 long array
            offset_locs = offset_locs.reshape(np.int(np.sum(offset_locs>0)/2),2) # reshape back to 2-column array
            offsets = offsets[offsets<0]
        
        offsets *= -1 # make them all positive
    
    # THIS IS FOR RIGHT LATERAL:   
    else: 
        if slip == 0:
            offset_locs = offset_locs[offsets>=0] # this turns it into 1 long array
            offset_locs = offset_locs.reshape(np.int(np.sum(offset_locs>0)/2),2) # reshape back to 2-column array
            offsets = offsets[offsets>=0]
        else:
            offset_locs = offset_locs[offsets>0] # this turns it into 1 long array
            offset_locs = offset_locs.reshape(np.int(np.sum(offset_locs>0)/2),2) # reshape back to 2-column array
            offsets = offsets[offsets>0]
    
    # scale offsets by grid size
    offsets = offsets * dxy
    

    
    ########################   
    ### save offset data ###
    ########################
    pickle.dump([offsets,offset_locs],open('%s/offsets_%s_%s.p' %(save_loc,name,years),'wb'))
    
    ################################   
    ### plot channels w/ offsets ###
    ################################    
    
    if plot:
        y_above = np.zeros_like(offsets)
        y_below = np.zeros_like(offsets)
        y_above[:] = fault_loc-fzw
        y_below[:] = fault_loc+fzw
        
        fig = plt.subplots(figsize=(12,8))
        plt.imshow(da,cmap='Greys',vmin=0,vmax=1000)
        plt.imshow(channels,cmap='winter') # rainbow gives purple, raibow_r gives red
        plt.axhline(fault_loc+fzw,0,1000,color='r',linestyle='--')
        plt.axhline(fault_loc-fzw,0,1000,color='r',linestyle='--')
        plt.scatter(offset_locs[:,0],y_above)#c=offsets)
        plt.scatter(offset_locs[:,1],y_below)#c=offsets)
        plt.savefig('%s/channels_%s_%s.png' %(save_loc,name,years),dpi=600,facecolor=(1,1,1,0))
        plt.show()
        plt.clf()

    #######################    
    ### calculate stats ###
    #######################
    
    # calculate mean
    mean = np.round(np.mean(offsets),2)
    #print('mean: %s' %mean)
    
    # calculate rmse
    real_offsets = np.zeros_like(offsets)
    real_offsets[:]=real_max_slip
    rmse = np.round(np.sqrt(mean_squared_error(real_offsets, offsets)),2)
    #print('RMSE: %s' %rmse)
    
    # calculate standard deviation
    std = np.round(np.std(offsets),2)
    #print('StDev: %s' %std)
    
    # make PDFs 
    offsets_sorted = np.sort(offsets)
    pdf = norm.pdf(offsets_sorted, mean, std)
    
    #######################    
    ### clean offsets?? ###
    #######################  
    if clean:
        # throw out outliers = any >or< than 2*std
        offsets_clean = np.copy(offsets)
        offsets_clean[offsets>(mean+2*std)] = np.nan
        offsets_clean[offsets_clean<(mean-2*std)] = np.nan
        
        # also save locations of the clean set of offsets --> ugh do later
        offset_locs_clean = np.copy(offset_locs)
        offset_locs_clean = offset_locs_clean[offsets_clean>0]
        
        # remove nan entries       
        offsets_clean = offsets_clean[offsets_clean>0]
        
        # then recalculate stats
        # calculate mean
        mean_clean = np.round(np.mean(offsets_clean),2)
        #print('mean: %s' %mean)
        
        # calculate rmse
        real_offsets_clean = np.zeros_like(offsets_clean)
        real_offsets_clean[:]=real_max_slip    
        rmse_clean = np.round(np.sqrt(mean_squared_error(real_offsets_clean, offsets_clean)),2)
        #print('RMSE: %s' %rmse)
        
        # calculate standard deviation
        std_clean = np.round(np.std(offsets_clean),2)
        #print('StDev: %s' %std)
        
        # make PDFs 
        offsets_sorted_clean = np.sort(offsets_clean)
        pdf_clean = norm.pdf(offsets_sorted_clean, mean_clean, std_clean)
        
   
    ########################   
    ### plot offset data ###
    ########################
    if plot:
        
        ########################### 
        ### displacement vs fzw ###
        ########################### 
        x = np.zeros_like(offsets)
        x[:] = fzw
        
        fig,ax = plt.subplots(figsize=(8,6))
        plt.axhline(real_offsets[0],color='indianred',linestyle='--')
        
        plt.errorbar(x,offsets,yerr=std,lw=1,marker='o',linestyle=None,color='royalblue')#mfc='k', mec='black',ecolor='k' )#ms=4, mew=2)
        plt.plot(pdf+fzw, offsets_sorted,'-',color='royalblue')
        plt.fill_betweenx(offsets_sorted,fzw,pdf+fzw,alpha=.3,color='royalblue')
        plt.text(0.02,.95,'mean: %s'%mean,transform=ax.transAxes,color='royalblue')
        plt.text(0.02,.90,'RMSE: %s'%rmse,transform=ax.transAxes,color='royalblue')
        plt.text(0.02,.85,'std: %s'%std,transform=ax.transAxes,color='royalblue')
        
        if clean:
            x_clean = np.zeros_like(offsets_clean)
            x_clean[:] = fzw
            #plt.plot(x_clean,offsets_clean,'o',color='seagreen')
            plt.errorbar(x_clean,offsets_clean,yerr=std_clean,lw=1,marker='o',linestyle=None,color='k')#mfc='k', mec='black',ecolor='k' )#ms=4, mew=2)
            plt.plot(pdf_clean+fzw, offsets_sorted_clean,ls='--',color='k')
            plt.fill_betweenx(offsets_sorted_clean,fzw,pdf_clean+fzw,linestyle='--',alpha=.3,color='k')
            plt.text(0.2,.95,'mean clean: %s'%mean_clean,transform=ax.transAxes)
            plt.text(0.2,.90,'RMSE clean: %s'%rmse_clean,transform=ax.transAxes)
            plt.text(0.2,.85,'std clean: %s'%std_clean,transform=ax.transAxes)
        
        
        plt.text(fzw-.45,real_max_slip+1,'actual offset',color='indianred')
        plt.xlim(fzw-.5,fzw+.5)
        plt.ylabel('apparent offset measurement (m)')
        plt.xlabel('fault zone width (m)')
        plt.savefig('%s/offsets_%s_%s.png'%(save_loc,name,years),dpi=450,facecolor=(1,1,1,0))
        #plt.show()
        plt.clf()
        
        #################################
        ### displacement along strike ###
        #################################
        # x location of each offset is mean of above and below thalwag indices
        x = np.zeros_like(offsets)
        for i in range(len(offset_locs)):
            x[i] = (offset_locs[i,0]+offset_locs[i,1]) / 2.
        
        fig,ax = plt.subplots(figsize=(6,4))
        plt.fill_between((0,1000),mean+std,mean-std,alpha=0.1,color='k')         
        plt.axhline(real_offsets[0],color='indianred',linestyle='--',linewidth=1,label='real')
        plt.axhline(mean,color='k',linestyle='-.',linewidth=1,label='mean')
        plt.plot(x,offsets,'.')
        if clean: 
            plt.axhline(mean_clean,linestyle='-',linewidth=1,label='clean')
        plt.xticks((0,100,200,300,400,500,600,700,800,900,1000))
        plt.yticks((0,5,10,15,20,25,30,35,40,45,50,55,60))
        plt.xlabel('distance along strike (m)')
        plt.ylabel('apparent offset (m)')
        #plt.grid(which='both',color='lightgray',linewidth=.5,linestyle='--')
        plt.xlim(0,1000)
        plt.ylim(-3,60)
        plt.legend(loc='upper right',fontsize=10)
        plt.savefig('%s/offsets_along_strike_%s_%s.png' %(save_loc,name,years),dpi=300,facecolor=(1,1,1,0))
        plt.show()
    
        

    return offsets, offset_locs, thal_index_above, thal_index_below
    


#%% run for 1 model - measure_offsets

# load parameter configuration from config.yaml file - yaml not playing nice with gdal
#config = yaml.load(open('config_files/config.yaml','r')) # use this to run fron config/here
#config = yaml.load(open(sys.argv[1],'r')) # use this line to read file from command line

#model_name = config['saving']['model_name'] # model name for naming outputs - ztopo, avgZ, displacement, movie
#dxy = config['grid']['dxy'] # grid step in meters
#total_slip = config['strike-slip']['total_slip'] # total slip for entire model time [meters]
#tmax = config['time']['tmax'] # total time in years

model_name = 'fzw0'
dxy = 1
total_slip = 30
tmax = 10000

output_loc = 'model_output/%s' %model_name # location to save output
da_ascii = '%s/da_%s.asc' %(output_loc,model_name)
profile_distance = 10 # number of meters away from fault on either side to measure thalweg offset distances


cutoff = 1000 # drainage area threshold 
plot=True
clean=True # this is for plotting purposes only!

offsets, offset_locs, thal_index_above, thal_index_below = measure_offsets(da_ascii,profile_distance,dxy,total_slip,model_name,save_loc=output_loc,years=tmax,channel_cutoff=cutoff,clean=clean,plot=plot)
