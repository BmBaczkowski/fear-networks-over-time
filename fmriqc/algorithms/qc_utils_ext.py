# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:11:02 2017

@author: baczkowski
"""

import math
import os
import time
import nibabel as nb
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from pylab import cm
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
from nipy.labs import viz
import pylab as plt
import matplotlib.pyplot as pyplt
import matplotlib.patches as mpatches

#functions for plotting distributions
def plot_vline(cur_val, label, ax):
    ax.axvline(cur_val)
    ylim = ax.get_ylim()
    vloc = (ylim[0] + ylim[1]) / 2.0
    xlim = ax.get_xlim()
    pad = (xlim[0] + xlim[1]) / 100.0
    ax.text(cur_val - pad, vloc, label, color="blue", rotation=90, verticalalignment='center', horizontalalignment='right')

def _get_values_inside_a_mask(main_file, mask_file):
    main_nii = nb.load(main_file)
    main_data = main_nii.get_data()
    nan_mask = np.logical_not(np.isnan(main_data))
    mask = nb.load(mask_file).get_data() > 0
    
    data = main_data[np.logical_and(nan_mask, mask)]
    return data

def get_median_distribution(main_files, mask_files):
    medians = []
    for main_file, mask_file in zip(main_files, mask_files):
	#print main_file
        med = np.median(_get_values_inside_a_mask(main_file, mask_file))
        medians.append(med)
    return medians

def plot_distrbution_of_values(main_file, mask_file, xlabel, distribution=None, xlabel2=None, figsize=(11.7,8.3)):
    data = _get_values_inside_a_mask(main_file, mask_file)

    fig = Figure(figsize=figsize)
    FigureCanvas(fig)
    
    gs = GridSpec(2, 1)
    ax = fig.add_subplot(gs[0, 0])
    sns.distplot(np.array(data, dtype=np.double), kde=False, bins=100, ax=ax) #sns.distplot(data.astype(np.double), kde=False, bins=100, ax=ax)
    ax.set_xlabel(xlabel)
    
    ax = fig.add_subplot(gs[1, 0])
    sns.distplot(np.array(distribution, dtype=np.double), ax=ax) #sns.distplot(np.array(distribution).astype(np.double), ax=ax)
    cur_val = np.median(data)
    label = "%g"%cur_val
    plot_vline(cur_val, label, ax=ax)
    ax.set_xlabel(xlabel2)
    
    return fig
    
    
def get_mean_frame_displacement_disttribution(FD_files):
    mean_FDs = []
    max_FDs = []
    for FD_file in FD_files:
        FD_power = np.loadtxt(FD_file)
        mean_FDs.append(FD_power.mean())
        max_FDs.append(FD_power.max())
        
    return mean_FDs, max_FDs

def plot_frame_displacement(FD_file, mean_FD_distribution=None, figsize=(11.7,8.3)):

    FD_power = np.loadtxt(FD_file)

    fig = Figure(figsize=figsize)
    FigureCanvas(fig)
    
    if mean_FD_distribution:
        grid = GridSpec(2, 4)
    else:
        grid = GridSpec(1, 4)
    
    ax = fig.add_subplot(grid[0,:-1])
    ax.plot(FD_power)
    ax.set_xlim((0, len(FD_power)))
    ax.set_ylabel("Frame Displacement [mm]")
    ax.set_xlabel("Frame number")
    ylim = ax.get_ylim()
    
    ax = fig.add_subplot(grid[0,-1])
    sns.distplot(FD_power, vertical=True, ax=ax)
    ax.set_ylim(ylim)
    
    if mean_FD_distribution:
        ax = fig.add_subplot(grid[1,:])
        sns.distplot(mean_FD_distribution, ax=ax)
        ax.set_xlabel("Mean Frame Displacement (over all subjects) [mm]")
        MeanFD = FD_power.mean()
        label = "MeanFD = %g"%MeanFD
        plot_vline(MeanFD, label, ax=ax)
        
    fig.suptitle('motion', fontsize='14')
        
    return fig

def get_mean_DVARS_disttribution(DVARS_files):
    mean_DVARS = []
    max_DVARS = []
    for DVARS_file in DVARS_files:
        DVARS = np.loadtxt(DVARS_file)
        mean_DVARS.append(DVARS.mean())
        max_DVARS.append(DVARS.max())
        
    return mean_DVARS, max_DVARS

def plot_DVARS(title, DVARS_file, mean_DVARS_distribution=None, figsize=(11.7,8.3)):

    DVARS = np.loadtxt(DVARS_file)

    fig = Figure(figsize=figsize)
    FigureCanvas(fig)
    
    if mean_DVARS_distribution:
        grid = GridSpec(2, 4)
    else:
        grid = GridSpec(1, 4)
    
    ax = fig.add_subplot(grid[0,:-1])
    ax.plot(DVARS)
    ax.set_xlim((0, len(DVARS)))
    ax.set_ylabel("DVARS")
    ax.set_xlabel("Frame number")
    ylim = ax.get_ylim()
    
    ax = fig.add_subplot(grid[0,-1])
    sns.distplot(DVARS, vertical=True, ax=ax)
    ax.set_ylim(ylim)
    
    if mean_DVARS_distribution:
        ax = fig.add_subplot(grid[1,:])
        sns.distplot(mean_DVARS_distribution, ax=ax)
        ax.set_xlabel("Mean DVARS (over all subjects) [std]")
        MeanFD = DVARS.mean()
        label = "Mean DVARS = %g"%MeanFD
        plot_vline(MeanFD, label, ax=ax)
        
    fig.suptitle(title, fontsize='14')
        
    return fig
    
def get_similarity_distribution(mincost_files):
    similarities = []
    for mincost_file in mincost_files:
        similarity = float(open(mincost_file, 'r').readlines()[0].split()[0])
        similarities.append(similarity)
    return similarities

def plot_epi_T1_corregistration(mean_epi_file, wm_file, subject_id, similarity_distribution=None, figsize=(11.7,8.3),):
       
    fig = plt.figure(figsize=figsize)
    
    if similarity_distribution:
        ax = plt.subplot(2,1,1)
        sns.distplot(similarity_distribution.values(), ax=ax)
        ax.set_xlabel("EPI-T1 mincost function (over all subjects)")
        cur_similarity = similarity_distribution[subject_id]
        label = "mincost function = %g"%cur_similarity
        plot_vline(cur_similarity, label, ax=ax)
        
        ax = plt.subplot(2,1,2)
    else:
        ax = plt.subplot(1,1,0)
    
  

    func = nb.load(mean_epi_file).get_data()
    func_affine = nb.load(mean_epi_file).get_affine()
    
    wm_data = nb.load(wm_file).get_data()
    wm_affine = nb.load(wm_file).get_affine()
    
    slicer = viz.plot_anat(np.asarray(func), np.asarray(func_affine), black_bg=True,
                           cmap = cm.Greys_r,  # @UndefinedVariable
                           figure = fig,
                           axes = ax,
                           draw_cross = False)
    slicer.contour_map(np.asarray(wm_data), np.asarray(wm_affine), linewidths=[0.1], colors=['r',])
    
    fig.suptitle('coregistration', fontsize='14')
    
    return fig

def _calc_rows_columns(ratio, n_images):
    rows = 1
    for _ in range(100):
        columns = math.floor(ratio * rows)
        total = rows * columns
        if total > n_images:
            break

        columns = math.ceil(ratio * rows)
        total = rows * columns
        if total > n_images:
            break
        rows += 1
    return rows, columns

def plot_mosaic(nifti_file, title=None, overlay_mask = None, figsize=(11.7,8.3)):
    if isinstance(nifti_file,str): 
        nii = nb.load(nifti_file)
        mean_data = nii.get_data()
    else:
        mean_data = nifti_file
   
    n_images = mean_data.shape[2]
    row, col = _calc_rows_columns(figsize[0]/figsize[1], n_images)
    
    if overlay_mask:
        overlay_data = nb.load(overlay_mask).get_data()

    # create figures
    fig = Figure(figsize=figsize)
    FigureCanvas(fig)
    
    fig.subplots_adjust(top=0.85)
    for image in (range(n_images)):
        ax = fig.add_subplot(row, col, image+1)
        data_mask = np.logical_not(np.isnan(mean_data))
        if overlay_mask:
            ax.set_rasterized(True)
        ax.imshow(np.fliplr(mean_data[:,:,image].T), vmin=np.percentile(mean_data[data_mask], 0.5), 
                   vmax=np.percentile(mean_data[data_mask],99.5), 
                   cmap=cm.Greys_r, interpolation='nearest', origin='lower')  # @UndefinedVariable
        if overlay_mask:
            cmap = cm.Reds  # @UndefinedVariable
            cmap._init() 
            alphas = np.linspace(0, 0.75, cmap.N+3)
            cmap._lut[:,-1] = alphas
            ax.imshow(np.fliplr(overlay_data[:,:,image].T), vmin=0, vmax=1,
                   cmap=cmap, interpolation='nearest', origin='lower')  # @UndefinedVariable
            
        ax.axis('off')
    fig.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.95, wspace=0.01, hspace=0.1)
    
    if not title:
        _, title = os.path.split(nifti_file)
        title += " (last modified: %s)"%time.ctime(os.path.getmtime(nifti_file))
    fig.suptitle(title, fontsize='14')
    
    return fig    
    
    
#################### POWER PLOTS ##############################################

def plot_power(subject_id, session,
               FD_file, 
               onsets,
               data_reshaped_sorted, 
               hline_wm, hline_subcort, 
               data_post_reshaped_sorted,
               DVARSpre_file, 
               DVARSpost_file,
               GS_pre,
               GS_post,
               SCR,
               seq,
               nvol,
               figsize=(14,8)):
    
    FD = np.loadtxt(FD_file)
    DVARSpre = np.loadtxt(DVARSpre_file)
    DVARSpost = np.loadtxt(DVARSpost_file)

    
    sns.set_style('white')    
    sns.set(font_scale=.8)
    
    
    fig = pyplt.figure(1)
    fig.set_size_inches=figsize
    fig.subplots_adjust(hspace=.25)
    
    
    ax1 = pyplt.subplot2grid((5,3), (0,0), colspan=3)
    sns.tsplot(FD, range(1,nvol+1), color='k', ax=ax1, linewidth=0.8)
    [ax1.axvspan(x, x+1, facecolor='b', alpha=.5, fill=True) for x in onsets['CSminus']]
    [ax1.axvspan(x, x+1, facecolor='g', alpha=.5, fill=True) for x in onsets['CSplus']]
    p1 = mpatches.Patch(color='b', label='CS-')
    p2 = mpatches.Patch(color='g', label='CS+')
    ax1.legend(handles=[p1,p2], bbox_to_anchor=(1, 1), loc=2)
    ax1.set_xticks([])
    ax1.set_ylabel('FD (mm)')
    ax1.set_title('Head motion')
    ax1.set_xlim([1,nvol])
    
    
    ax2 = pyplt.subplot2grid((5,3), (1,0), colspan=3)
    sns.tsplot(DVARSpre, range(2,nvol+1), color='b', ax=ax2, linewidth=0.8)
    sns.tsplot(DVARSpost-1, range(2,nvol+1), color='r', ax=ax2, linewidth=0.8)
    ax2.set_ylabel('DVARS (std)')
    ax2.set_title('BOLD')
    ax2.yaxis.tick_right()
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim([2,nvol])
    ax2.legend(['Pre', 'Post'], bbox_to_anchor=(1, 1), loc=2)
    
    ax3 = pyplt.subplot2grid((5,3), (2,0), colspan=3)
    sns.tsplot(GS_pre, range(1,nvol+1), color='g', ax=ax3, linewidth=0.8)
    sns.tsplot(GS_post, range(1,nvol+1), color='y', ax=ax3, linewidth=0.8)
    if subject_id != '25' or session != 'EXT2':
        sns.tsplot(SCR-np.max(SCR)-.8, seq, color='b', ax=ax3, marker='o', linewidth=0.5, markersize=2)
    #ax3.set_ylabel('BOLD')
    ax3.set_title('Global signal & SCR amplitudes')
    ax3.yaxis.tick_right()
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlim([1,nvol])
    ax3.legend(['GS Pre', 'GS Post'], bbox_to_anchor=(1, 1), loc=2)
    
#    m = np.mean(data_reshaped_sorted/100)
#    std = np.std(data_reshaped_sorted/100)
    
    ax4 = pyplt.subplot2grid((5,3), (3,0), colspan=3)
    ax4.figure.set_size_inches=(12,8)
    ax4.imshow(data_reshaped_sorted/100, 
               interpolation='nearest', 
               aspect = 'auto', 
               #vmin=m-3*std, vmax=m+3*std,
               vmin=-20, vmax=20,
               cmap='gray')
    ax4.set_yticks([])
    ax4.axhline(hline_subcort, color='b',  linewidth=.8) 
    ax4.axhline(hline_wm, color='y',  linewidth=.8)
    ax4.set_xticks([])
    ax4.set_title('(Sub-)Cortex & WM voxel signals (minimal pre-processing)')
    ax4.grid(False)
    ax4.set_xlim([1,nvol])
    
    
#    m = np.mean(data_post_reshaped_sorted/100)
#    std = np.std(data_post_reshaped_sorted/100)
    
    ax5 = pyplt.subplot2grid((5,3), (4,0), colspan=3)
    ax5.figure.set_size_inches=(12,8)
    ax5.imshow(data_post_reshaped_sorted/100, 
               interpolation='nearest', 
               aspect = 'auto', 
               #vmin=m-3*std, vmax=m+3*std,
               vmin=-20, vmax=20,
               cmap='gray')
    ax5.set_yticks([])
    ax5.axhline(hline_subcort, color='b',  linewidth=.8) 
    ax5.axhline(hline_wm, color='y',  linewidth=.8)
    ax5.set_xticks([1,nvol])
    ax5.set_title('(Sub-)Cortex & WM voxel signals (post-processing)')
    ax5.set_xlabel('Volume # (TR=1.96s)')
    ax5.grid(False)
    ax5.set_xlim([1,nvol])
    
    fig.suptitle('Power plots', fontsize='14')    
    
    return fig

































    
    
    
    
    