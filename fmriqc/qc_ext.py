# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:59:14 2017

@author: baczkowski
based on https://github.com/juhuntenburg/mriqc
"""
import os
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
from nipype import config


dynorphin_dir = '/nobackup/usbekistan4/baczkowski/DYNORPHIN'
data_dir = os.path.join(dynorphin_dir, 'sink_dir')
wd = os.path.join(dynorphin_dir, 'working_dir/')
sink = os.path.join(dynorphin_dir, 'sink_dir/QC/')

subjects = [ 
            '01', 
            '02', '03', '04', '05', '06', '07', '08', '09',
            '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
            '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
            '30', '31', '32', '33', '34']
            
sessions=['EXT1', 'EXT2']
#sessions=['EXT2']

###############################################################################
wf = pe.Workflow(name='QC')
wf.base_dir = wd
wf.config['execution']['crashdump_dir'] = wf.base_dir + "/crash_files"

nipype_cfg = dict(logging=dict(workflow_level='DEBUG'), execution={'stop_on_first_crash': False,
                                                                   'remove_unnecessary_outputs': False,
                                                                   'job_finished_timeout': 120})
config.update_config(nipype_cfg)


#generate distributions per session
subjects2 = ['01', 
            '02', '03', '04', '05', '06', '07', '08', '09',
            '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
            '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
            '30', '31', '32', '33', '34']
            
def get_lists_of_files(data_dir, session, subjects):
    
    tsnr_files = [data_dir+ "/MPP/%s/%s/TEMP/moco_tSNR.nii.gz"%(subject, session) for subject in subjects]
    mask_files = [data_dir+ "/MPP/%s/%s/QC/func_data_mask2func.nii.gz"%(subject,session) for subject in subjects]
    FD_files = [data_dir+ "/MPP/%s/%s/QC/func_data_FD.txt"%(subject, session) for subject in subjects]
    DVARSpre_files = [data_dir+ "/DENOISE_aCompCor/%s/%s/QC/stdDVARS_pre.txt"%(subject, session) for subject in subjects]
    DVARSpost_files = [data_dir+ "/DENOISE_aCompCor/%s/%s/QC/stdDVARS_post.txt"%(subject, session) for subject in subjects]
    mincost_func_files = [data_dir+ "/MPP/%s/%s/COREG/func2anat.dat.mincost"%(subject, session) for subject in subjects]
    
    return tsnr_files, mask_files, FD_files, DVARSpre_files, DVARSpost_files, mincost_func_files

def get_distributions(subjects, tsnr_files, mask_files, FD_files, DVARSpre_files, DVARSpost_files, mincost_func_files):
    
    from algorithms.qc_utils_ext import get_median_distribution
    from algorithms.qc_utils_ext import get_mean_frame_displacement_disttribution
    from algorithms.qc_utils_ext import get_mean_DVARS_disttribution
    from algorithms.qc_utils_ext import get_similarity_distribution
       
    
    
    tsnr_distribution = get_median_distribution(tsnr_files, mask_files)
    mean_FD_distribution, max_FD_distribution = get_mean_frame_displacement_disttribution(FD_files)
    mean_DVARSpre_distribution, max_DVARSpre_distribution = get_mean_DVARS_disttribution(DVARSpre_files)
    mean_DVARSpost_distribution, max_DVARSpost_distribution = get_mean_DVARS_disttribution(DVARSpost_files)
    similarity_distribution = get_similarity_distribution(mincost_func_files)
    similarity_distribution = dict(zip(subjects, similarity_distribution))

    return tsnr_distribution, mean_FD_distribution, mean_DVARSpre_distribution, mean_DVARSpost_distribution, similarity_distribution
    
#infosource to interate over sessions: EXT1, EXT2
sessions_infosource = pe.Node(util.IdentityInterface(fields=['session']),name='session')
sessions_infosource.iterables = [('session', sessions)]  

run_get_lists_of_files = pe.Node(util.Function(input_names=['data_dir', 'session', 'subjects'],#, 'sessions'],
                                               output_names=['tsnr_files', 'mask_files', 'FD_files', 'DVARSpre_files', 'DVARSpost_files', 'mincost_func_files'], #'filelist'
                                               function=get_lists_of_files), 
                                               name='run_get_lists_of_files')

run_get_lists_of_files.inputs.data_dir = data_dir
run_get_lists_of_files.inputs.subjects = subjects2

wf.connect(sessions_infosource, 'session', run_get_lists_of_files, 'session')

run_get_distributions = pe.Node(util.Function(input_names=['subjects', 'tsnr_files', 'mask_files', 'FD_files', 'DVARSpre_files', 'DVARSpost_files', 'mincost_func_files'],
                                               output_names=['tsnr_distribution', 'mean_FD_distribution', 'mean_DVARSpre_distribution', 'mean_DVARSpost_distribution', 'similarity_distribution'], 
                                               function=get_distributions), 
                                               name='run_get_distributions')

run_get_distributions.inputs.subjects = subjects2
wf.connect(run_get_lists_of_files, 'tsnr_files', run_get_distributions, 'tsnr_files')
wf.connect(run_get_lists_of_files, 'mask_files', run_get_distributions, 'mask_files')
wf.connect(run_get_lists_of_files, 'FD_files', run_get_distributions, 'FD_files')
wf.connect(run_get_lists_of_files, 'DVARSpre_files', run_get_distributions, 'DVARSpre_files')
wf.connect(run_get_lists_of_files, 'DVARSpost_files', run_get_distributions, 'DVARSpost_files')
wf.connect(run_get_lists_of_files, 'mincost_func_files', run_get_distributions, 'mincost_func_files')

                                           
                                               
#infosource to interate over subjects
subjects_infosource = pe.Node(util.IdentityInterface(fields=['subject']),name='subject')
subjects_infosource.iterables = [('subject', subjects)]  

#select files
templates = {'mean_epi_file':         'MPP/{subject}/{session}/TEMP/moco_Tmean.nii.gz',
             'mask_file':             'MPP/{subject}/{session}/QC/func_data_mask2func.nii.gz',
             'tsnr_file':             'MPP/{subject}/{session}/TEMP/moco_tSNR.nii.gz',
             'FD_file':               'MPP/{subject}/{session}/QC/func_data_FD.txt',
             'wmedge_file':           'fs_out/{subject}/T1_brain_wmedge.nii.gz',
             'epi2anat_coreg_file':   'MPP/{subject}/{session}/TEMP/moco_Tmean_restored2anat.nii.gz',
             'data_pre_file':         'MPP/{subject}/{session}/prefiltered_func_data_detrend.nii.gz',
             'data_post_file':        'DENOISE_aCompCor/{subject}/{session}/filtered_func_data_hp01.nii.gz',
             'GM_msk_file':           'MASKS/{subject}/aparc_asec.GM_RIBBONmaskEPI.nii.gz',
             'WM_msk_file':           'MASKS/{subject}/aparc_asec.WMmask_ero2EPI.nii.gz',
             'SBCORT_msk_file':       'MASKS/{subject}/aparc_asec.GM_SCmaskEPI.nii.gz',
             'INBRAIN_msk_file':      'MASKS/{subject}/aparc_asec.INBRAINmaskEPI.nii.gz',
             'DVARSpre_file':         'DENOISE_aCompCor/{subject}/{session}/QC/stdDVARS_pre.txt',
             'DVARSpost_file':        'DENOISE_aCompCor/{subject}/{session}/QC/stdDVARS_post.txt',
             'SCR_file':              'EDA/{session}/SCR_STIM/{subject}_SCR_seq.txt'}

selectfiles = pe.Node(nio.SelectFiles(templates, 
                                    base_directory=data_dir),
                                    name='selectfiles')


ds = pe.Node(nio.DataSink(base_directory=sink,
                          parameterization=False),
                          name='sink')  

wf.connect(subjects_infosource, 'subject', selectfiles, 'subject')
wf.connect(sessions_infosource, 'session', selectfiles, 'session') 
wf.connect(sessions_infosource, 'session', ds, 'container') 


########## main function #####################################################
def create_report(subject_id, 
                  mean_epi_file, 
                  mask_file, 
                  tsnr_file, 
                  tsnr_distribution,
                  FD_file,
                  mean_FD_distribution,
                  wmedge_file,
                  epi2anat_coreg_file,
                  similarity_distribution,
                  data_pre_file,
                  data_post_file,
                  GM_msk_file,
                  WM_msk_file,
                  SBCORT_msk_file,
                  INBRAIN_msk_file,
                  DVARSpre_file,
                  DVARSpost_file,   
                  SCR_file,
                  nvol,
                  session,
                  onsets):

    import gc
    import os
    import pylab as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from algorithms.qc_utils_ext import plot_mosaic, plot_distrbution_of_values
    from algorithms.qc_utils_ext import plot_epi_T1_corregistration
    from algorithms.qc_utils_ext import plot_frame_displacement
    from algorithms.qc_utils_ext import plot_power
    import numpy as np
    import nibabel as nb

    
    #demean fMRI data 
    def demean(infile):
        data = nb.load(infile).get_data().astype('float64')
        
        for i in range(np.shape(data)[0]):
            for j in range(np.shape(data)[1]):
                for k in range(np.shape(data)[2]):
                    data[i,j,k,:] = data[i,j,k,:] - np.mean(data[i,j,k,:])
        
        return data
    
    #sort data according to mask compartments 
    def get_data_sorted(GM_msk_file, WM_msk_file, SBCORT_msk_file, data_demean, nvol):
        
        GM_RIBBONmsk = nb.load(GM_msk_file).get_data().astype('int')
        WMmsk = nb.load(WM_msk_file).get_data().astype('int')
        SUBCORTmsk = nb.load(SBCORT_msk_file).get_data().astype('int')
        
        
        msk = np.zeros(GM_RIBBONmsk.shape)
        msk[GM_RIBBONmsk==1] = 1
        msk[SUBCORTmsk==1] = 2
        msk[WMmsk==1] = 3
        
        msk_reshaped = np.reshape(msk,-1)
        data_reshaped = np.reshape(data_demean, (msk_reshaped.shape[0], nvol))
        
        #get rid of zeros
        data_reshaped = data_reshaped[msk_reshaped!=0]
        msk_reshaped = msk_reshaped[msk_reshaped!=0]
        
        #sorting
        idx = np.argsort(msk_reshaped)
        msk_reshaped_sorted = msk_reshaped[idx]
        data_reshaped_sorted = data_reshaped[idx]
        
        #border for the plot
        hline_wm = np.where(msk_reshaped_sorted==2)[0][0]
        hline_subcort = np.where(msk_reshaped_sorted==3)[0][0]
        
        return data_reshaped_sorted, hline_wm, hline_subcort
        
    #get global signal
    def get_GS(INBRAIN_msk_file, data):
        
        INBRAINmsk = nb.load(INBRAIN_msk_file).get_data().astype('int')
        GS = np.mean(data[INBRAINmsk==1], axis=0)
        GS = GS/100
        return GS
    
    def get_SCR(SCR_file):
        
        SCR = np.loadtxt(SCR_file)
        
        return SCR
        
    
    data_pre_demean = demean(data_pre_file)
    data_post_demean = demean(data_post_file)
    
    [data_pre_demean_sorted, hline_wm, hline_subcort] = get_data_sorted(GM_msk_file, WM_msk_file, SBCORT_msk_file, data_pre_demean, nvol)
    [data_post_demean_sorted, hline_wm, hline_subcort] = get_data_sorted(GM_msk_file, WM_msk_file, SBCORT_msk_file, data_post_demean, nvol)
    
    GSpre = get_GS(INBRAIN_msk_file, data_pre_demean)
    GSpost = get_GS(INBRAIN_msk_file, data_post_demean)
    
    SCR = get_SCR(SCR_file)
    seq = np.sort(np.hstack((onsets['CSminus'], onsets['CSplus'])))
    
    #output_file = '/nobackup/usbekistan4/baczkowski/DYNORPHIN/sink_dir/%s/raport_%s.pdf'%(session, subject_id) 
    output_file = os.path.join(os.getcwd(), '%s_QC_raport_%s.pdf'%(subject_id, session))
    with PdfPages(output_file) as report:
       
    
        fig = plot_mosaic(mean_epi_file, title="Mean EPI", figsize=(8.3, 11.7))
        report.savefig(fig, dpi=300)
        fig.clf()
    
        
        fig = plot_mosaic(mean_epi_file, "Brain mask", mask_file, figsize=(8.3, 11.7))
        report.savefig(fig, dpi=600)
        fig.clf()
        
        fig = plot_mosaic(tsnr_file, title="tSNR", figsize=(8.3, 11.7))
        report.savefig(fig, dpi=300)
        fig.clf()
        
        fig = plot_distrbution_of_values(tsnr_file, mask_file, 
            "Subject %s tSNR inside the mask" % subject_id, 
            tsnr_distribution, 
            "Median tSNR (over all subjects)", 
            figsize=(8.3, 8.3))
        report.savefig(fig, dpi=300)
        fig.clf()
        plt.close()
        
        fig = plot_frame_displacement(FD_file, mean_FD_distribution, figsize=(8.3, 8.3))
        report.savefig(fig, dpi=300)
        fig.clf()
        plt.close()
        
        fig = plot_epi_T1_corregistration(epi2anat_coreg_file, wmedge_file, subject_id, similarity_distribution, figsize=(8.3, 8.3))
        report.savefig(fig, dpi=300)
        fig.clf()
        plt.close()
        
        fig = plot_power(subject_id, session, FD_file, onsets, data_pre_demean_sorted, hline_wm, hline_subcort, data_post_demean_sorted, DVARSpre_file, DVARSpost_file,
                         GSpre, GSpost, SCR, seq, nvol)
        report.savefig(fig, dpi=300)
        fig.clf()
        plt.close()
     
        report.close
        gc.collect()
        plt.close()
    
    return output_file

def get_nvol(session):
    temp = {'EXT1': 330,
            'EXT2': 364}
    
    nvol = temp[session]
    return nvol

def get_timings(session):
    
    temp = {'EXT1': 
            {'CSminus': [12.29, 28.66, 36.85, 61.41, 69.59, 85.97 ,102.34, 118.71, 143.27, 159.65, 167.83, 184.21, 192.39, 208.77, 233.33, 249.70, 266.07 ,282.45, 298.82, 307.00],
             'CSplus': [4.10, 20.48, 45.03, 53.22, 77.78, 94.15, 110.53, 126.90, 135.09, 151.46, 176.02, 200.58, 216.95, 225.14, 241.51, 257.89, 274.26 ,290.63 ,315.19, 323.38]}, 
            'EXT2': 
            {'CSminus': [20.47, 45.03, 53.22, 69.59, 85.97, 102.34, 118.71, 126.90, 143.27, 167.83, 184.21, 192.39, 208.77, 225.14, 241.51, 249.70, 266.07, 282.45, 290.63, 307.01, 323.38, 339.75],
             'CSplus': [4.10, 12.29, 28.66, 36.85, 61.41, 77.78, 94.15, 110.53, 135.09, 151.46, 159.65, 176.02, 200.58, 216.95, 233.33, 257.89, 274.26, 298.82, 315.19, 331.57, 347.94, 356.13]}
             }  
    onsets = temp[session]
    return onsets

report = pe.Node(util.Function(input_names=['subject_id', 
                                            'mean_epi_file',
                                            'mask_file',
                                            'tsnr_file', 
                                            'tsnr_distribution',
                                            'FD_file',
                                            'mean_FD_distribution',
                                            'wmedge_file', 
                                            'epi2anat_coreg_file',
                                            'similarity_distribution',
                                            'data_pre_file',
                                            'data_post_file',
                                            'GM_msk_file',
                                            'WM_msk_file',
                                            'SBCORT_msk_file',
                                            'INBRAIN_msk_file',
                                            'DVARSpre_file',
                                            'DVARSpost_file',
                                            'SCR_file',
                                            'nvol',
                                            'session',
                                            'onsets'], 
                                output_names=['out'],
                                function = create_report), name='report')

wf.connect(subjects_infosource, 'subject', report, 'subject_id')
wf.connect(selectfiles, 'mean_epi_file', report, 'mean_epi_file')
wf.connect(selectfiles, 'mask_file', report, 'mask_file')
wf.connect(selectfiles, 'tsnr_file', report, 'tsnr_file')
wf.connect(run_get_distributions, 'tsnr_distribution', report, 'tsnr_distribution')
wf.connect(selectfiles, 'FD_file', report, 'FD_file')
wf.connect(run_get_distributions, 'mean_FD_distribution', report, 'mean_FD_distribution')
wf.connect(selectfiles, 'wmedge_file', report, 'wmedge_file')
wf.connect(selectfiles, 'epi2anat_coreg_file', report, 'epi2anat_coreg_file')
wf.connect(run_get_distributions, 'similarity_distribution', report, 'similarity_distribution')
wf.connect(selectfiles, 'data_pre_file', report, 'data_pre_file')
wf.connect(selectfiles, 'data_post_file', report, 'data_post_file')
wf.connect(selectfiles, 'GM_msk_file', report, 'GM_msk_file')
wf.connect(selectfiles, 'WM_msk_file', report, 'WM_msk_file')
wf.connect(selectfiles, 'SBCORT_msk_file', report, 'SBCORT_msk_file')
wf.connect(selectfiles, 'INBRAIN_msk_file', report, 'INBRAIN_msk_file')
wf.connect(selectfiles, 'DVARSpre_file', report, 'DVARSpre_file')
wf.connect(selectfiles, 'DVARSpost_file', report, 'DVARSpost_file')
wf.connect(selectfiles, 'SCR_file', report, 'SCR_file')
wf.connect(sessions_infosource, ('session', get_nvol), report, 'nvol')
wf.connect(sessions_infosource, 'session', report, 'session')
wf.connect(sessions_infosource, ('session', get_timings), report, 'onsets')

wf.connect(report, 'out', ds, '@out')
############# RUN ##############################

wf.write_graph(dotfilename='wf.dot', graph2use='colored', format='pdf', simple_form=True)
#wf.run(plugin='CondorDAGMan')
wf.run(plugin='MultiProc', plugin_args={'n_procs': 17})












