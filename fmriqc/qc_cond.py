# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:59:14 2017

@author: baczkowski
Based on https://github.com/juhuntenburg/mriqc
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

subjects = ['01',
            '02', '03', '04', '05', '06', '07', '08', '09',
            '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
            '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
            '30', '31', '32', '33', '34']
            
sessions=['COND']



###############################################################################
wf = pe.Workflow(name='QC')
wf.base_dir = wd
wf.config['execution']['crashdump_dir'] = wf.base_dir + "/crash_files"

nipype_cfg = dict(logging=dict(workflow_level='DEBUG'), execution={'stop_on_first_crash': False,
                                                                   'remove_unnecessary_outputs': False,
                                                                   'job_finished_timeout': 120})
config.update_config(nipype_cfg)


#generate distributions per session
subjects2 = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
            '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
            '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
            '30', '31', '32', '33', '34']
def get_lists_of_files(data_dir, session, subjects):
    
    tsnr_files = [data_dir+ "/MPP/%s/%s/TEMP/moco_tSNR.nii.gz"%(subject, session) for subject in subjects]
    mask_files = [data_dir+ "/MPP/%s/%s/QC/func_data_mask2func.nii.gz"%(subject,session) for subject in subjects]
    FD_files = [data_dir+ "/MPP/%s/%s/QC/func_data_FD.txt"%(subject, session) for subject in subjects]
    DVARSpre_files = [data_dir+ "/DENOISE/%s/%s/QC/stdDVARS_pre.txt"%(subject, session) for subject in subjects]
    DVARSpost_files = [data_dir+ "/DENOISE/%s/%s/QC/stdDVARS_postB.txt"%(subject, session) for subject in subjects]
    mincost_func_files = [data_dir+ "/MPP/%s/%s/COREG/func2anat.dat.mincost"%(subject, session) for subject in subjects]
    
    return tsnr_files, mask_files, FD_files, DVARSpre_files, DVARSpost_files, mincost_func_files

def get_distributions(subjects, tsnr_files, mask_files, FD_files, DVARSpre_files, DVARSpost_files, mincost_func_files):
    
    from algorithms.qc_utils_cond import get_median_distribution
    from algorithms.qc_utils_cond import get_mean_frame_displacement_disttribution
    from algorithms.qc_utils_cond import get_mean_DVARS_disttribution
    from algorithms.qc_utils_cond import get_similarity_distribution
       
    
    
    tsnr_distribution = get_median_distribution(tsnr_files, mask_files)
    mean_FD_distribution, max_FD_distribution = get_mean_frame_displacement_disttribution(FD_files)
    mean_DVARSpre_distribution, max_DVARSpre_distribution = get_mean_DVARS_disttribution(DVARSpre_files)
    mean_DVARSpost_distribution, max_DVARSpost_distribution = get_mean_DVARS_disttribution(DVARSpost_files)
    similarity_distribution = get_similarity_distribution(mincost_func_files)
    similarity_distribution = dict(zip(subjects, similarity_distribution))

    return tsnr_distribution, mean_FD_distribution, mean_DVARSpre_distribution, mean_DVARSpost_distribution, similarity_distribution
    
#infosource to interate over sessions: COND, EXT1, EXT2
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
             'wm_file':               'fs_out/{subject}/T1_brain_wmedge.nii.gz',
             'epi2anat':              'MPP/{subject}/{session}/TEMP/moco_Tmean_restored2anat.nii.gz',
             'data_pre':              'MPP/{subject}/{session}/prefiltered_func_data_detrend.nii.gz',
             'data_post':             'DENOISE/{subject}/{session}/filtered_func_data_B_hp.nii.gz',
             'GMmask':                'MASKS/{subject}/aparc_asec.GM_RIBBONmaskEPI.nii.gz',
             'WMmask':                'MASKS/{subject}/aparc_asec.WMmask_ero2EPI.nii.gz',
             'DVARSpre':              'DENOISE/{subject}/{session}/QC/stdDVARS_pre.txt',
             'DVARSpost':             'DENOISE/{subject}/{session}/QC/stdDVARS_postB.txt',
             'INBRAIN':               'MASKS/{subject}/aparc_asec.INBRAINmaskEPI.nii.gz',
             'subcortmask':           'MASKS/{subject}/aparc_asec.GM_SCmaskEPI.nii.gz',
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



def create_report(subject_id, mean_epi_file, mask_file, 
                  tsnr_file, 
                  tsnr_distribution,
                  FD_file,
                  mean_FD_distribution,
                  wm_file,
                  epi2anat,
                  similarity_distribution,
                  data_pre,
                  data_post,
                  GMmask,
                  WMmask,
                  DVARSpre_file,
                  DVARSpost_file,
                  INBRAIN,
                  SCR_file,
                  subcortmask):

    import gc
    import os
    import pylab as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from algorithms.qc_utils_cond import plot_mosaic, plot_distrbution_of_values
    from algorithms.qc_utils_cond import plot_epi_T1_corregistration
    from algorithms.qc_utils_cond import plot_frame_displacement
    from algorithms.qc_utils_cond import plot_power
    import numpy as np
    import nibabel as nb
    
    CSminus=[4.10, 20.46, 44.99, 69.53, 85.88, 110.42, 126.77, 159.49, 167.66, 192.20, 208.55, 241.27, 257.62, 273.98, 290.33, 314.87, 331.22, 355.76, 380.29, 404.83]
    CSplus=[28.64, 53.17, 61.35, 94.06, 102.24, 134.95, 143.13, 175.84, 184.02, 216.73, 224.91, 249.44, 265.80, 298.51, 323.05, 339.40, 347.58, 363.94, 388.47, 396.65] 
    shock=[12.28, 36.81, 77.71, 118.60, 151.31, 200.38, 233.09, 282.16, 306.69, 372.11]
    
    def demean(infile):
        
        
        data = nb.load(infile).get_data().astype('float64')
        
        for i in range(np.shape(data)[0]):
            for j in range(np.shape(data)[1]):
                for k in range(np.shape(data)[2]):
                    data[i,j,k,:] = data[i,j,k,:] - np.mean(data[i,j,k,:])
        
        return data
    
    def get_data_sorted(GM_file, WM_file, subcortmask, data_demean, nvol):
        
        GM_RIBBONmsk = nb.load(GM_file).get_data().astype('int')
        WMmsk = nb.load(WM_file).get_data().astype('int')
        SUBCORTmsk = nb.load(subcortmask).get_data().astype('int')
        
        
        msk = np.zeros(GM_RIBBONmsk.shape)
        msk[GM_RIBBONmsk==1] = 1
        msk[SUBCORTmsk==1] = 2
        msk[WMmsk==1] = 3
        
        msk_reshaped = np.reshape(msk,-1)
        data_reshaped = np.reshape(data_demean, (msk_reshaped.shape[0],nvol))
        
        #get rid of zeros
        data_reshaped = data_reshaped[msk_reshaped!=0]
        msk_reshaped = msk_reshaped[msk_reshaped!=0]
        
        #sorting
        idx = np.argsort(msk_reshaped)
        msk_reshaped_sorted = msk_reshaped[idx]
        data_reshaped_sorted = data_reshaped[idx]
        
        #border
        hline_wm = np.where(msk_reshaped_sorted==2)[0][0]
        hline_subcort = np.where(msk_reshaped_sorted==3)[0][0]
        
        return data_reshaped_sorted, hline_wm, hline_subcort
        
    def get_GS(INBRAIN, data):
        
        INBRAINmsk = nb.load(INBRAIN).get_data().astype('int')
        GS = np.mean(data[INBRAINmsk==1], axis=0)
        GS = GS/100
        return GS
    
    def get_SCR(SCR_file):
        
        SCR = np.loadtxt(SCR_file)
        
        return SCR
        
    
    data_pre_demean = demean(data_pre)
    data_post_demean = demean(data_post)
    
    [data_pre_demean_sorted, hline_wm, hline_subcort] = get_data_sorted(GMmask, WMmask, subcortmask, data_pre_demean, 413)
    [data_post_demean_sorted, hline_wm, hline_subcort] = get_data_sorted(GMmask, WMmask, subcortmask, data_post_demean, 413)
    
    GSpre = get_GS(INBRAIN, data_pre_demean)
    GSpost = get_GS(INBRAIN, data_post_demean)

    SCR = get_SCR(SCR_file)
    onsets = np.sort(np.hstack((shock, CSminus, CSplus)))
    
    #output_file = '/nobackup/usbekistan4/baczkowski/DYNORPHIN/sink_dir/%s/raport_%s.pdf'%(session, subject_id) 
    output_file = os.path.join(os.getcwd(), '%s_QC_raport_cond.pdf'%(subject_id))
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
        
        fig = plot_epi_T1_corregistration(epi2anat, wm_file, subject_id, similarity_distribution, figsize=(8.3, 8.3))
        report.savefig(fig, dpi=300)
        fig.clf()
        plt.close()
        
        fig = plot_power(FD_file, CSminus, CSplus, shock, data_pre_demean_sorted, hline_wm, hline_subcort, data_post_demean_sorted, DVARSpre_file, DVARSpost_file,
                         GSpre, GSpost, SCR, onsets)
        report.savefig(fig, dpi=300)
        fig.clf()
        plt.close()
     
        report.close
        gc.collect()
        plt.close()
    
    return output_file


report = pe.Node(util.Function(input_names=['subject_id', 
                                            'mean_epi_file',
                                            'mask_file',
                                            'tsnr_file', 
                                            'tsnr_distribution',
                                            'FD_file',
                                            'mean_FD_distribution',
                                            'wm_file', 
                                            'epi2anat',
                                            'similarity_distribution',
                                            'data_pre',
                                            'data_post',
                                            'GMmask',
                                            'WMmask',
                                            'DVARSpre_file',
                                            'DVARSpost_file',
                                            'INBRAIN',
                                            'SCR_file',
                                            'subcortmask'], 
                                output_names=['out'],
                                function = create_report), name='report')

wf.connect(subjects_infosource, 'subject', report, 'subject_id')
wf.connect(selectfiles, 'mean_epi_file', report, 'mean_epi_file')
wf.connect(selectfiles, 'mask_file', report, 'mask_file')
wf.connect(selectfiles, 'tsnr_file', report, 'tsnr_file')
wf.connect(selectfiles, 'FD_file', report, 'FD_file')
wf.connect(selectfiles, 'wm_file', report, 'wm_file')
wf.connect(selectfiles, 'epi2anat', report, 'epi2anat')
wf.connect(run_get_distributions, 'tsnr_distribution', report, 'tsnr_distribution')
wf.connect(run_get_distributions, 'mean_FD_distribution', report, 'mean_FD_distribution')
wf.connect(run_get_distributions, 'similarity_distribution', report, 'similarity_distribution')
wf.connect(selectfiles, 'data_pre', report, 'data_pre')
wf.connect(selectfiles, 'data_post', report, 'data_post')
wf.connect(selectfiles, 'GMmask', report, 'GMmask')
wf.connect(selectfiles, 'WMmask', report, 'WMmask')
wf.connect(selectfiles, 'DVARSpre', report, 'DVARSpre_file')
wf.connect(selectfiles, 'DVARSpost', report, 'DVARSpost_file')
wf.connect(selectfiles, 'INBRAIN', report, 'INBRAIN')
wf.connect(selectfiles, 'SCR_file', report, 'SCR_file')
wf.connect(selectfiles, 'subcortmask', report, 'subcortmask')


wf.connect(report, 'out', ds, '@out')
############# RUN ##############################

wf.write_graph(dotfilename='wf.dot', graph2use='colored', format='pdf', simple_form=True)
#wf.run(plugin='CondorDAGMan')
wf.run(plugin='MultiProc', plugin_args={'n_procs': 17})












