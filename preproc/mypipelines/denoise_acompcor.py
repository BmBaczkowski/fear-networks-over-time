# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 21:19:19 2016

@author: baczkowski
"""

import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
from algorithms.dvars import compute_dvars 
from algorithms.motion import calc_friston_twenty_four
from algorithms.trends import calc_trends
from algorithms.CompCor import calc_compcor


def denoise(subject,
            sessions,
            data_dir, 
            wd,
            sink,
            TR):
                         
                         
    #initiate min func preproc workflow
    wf = pe.Workflow(name='DENOISE_aCompCor')
    wf.base_dir = wd
    wf.config['execution']['crashdump_dir'] = wf.base_dir + "/crash_files"
    
    ## set fsl output type to nii.gz
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    
    # I/O nodes
    inputnode = pe.Node(util.IdentityInterface(fields=['subjid']), 
                                                name='inputnode')  
    inputnode.inputs.subjid = subject

    
    ds = pe.Node(nio.DataSink(base_directory=sink,
                              parameterization=False),
                              name='sink')    

    
    #infosource to interate over sessions: COND, EXT1, EXT2
    sessions_infosource = pe.Node(util.IdentityInterface(fields=['session']),name='session')
    sessions_infosource.iterables = [('session', sessions)]  
    
    #select files
    templates = {'prefiltered':         'MPP/{subject}/{session}/prefiltered_func_data.nii.gz',
                 'prefiltered_detrend': 'MPP/{subject}/{session}/prefiltered_func_data_detrend.nii.gz',
                 'prefiltered_detrend_Tmean': 'MPP/{subject}/{session}/QC/prefiltered_func_data_detrend_Tmean.nii.gz',
                 'prefiltered_mask':    'MPP/{subject}/{session}/prefiltered_func_data_mask.nii.gz',
                 'WM_msk':              'MASKS/{subject}/aparc_asec.WMmask_ero2EPI.nii.gz',
                 'CSF_msk':             'MASKS/{subject}/aparc_asec.CSFmask_ero0EPI.nii.gz',
                 'motion_par':          'MPP/{subject}/{session}/MOCO/func_data_stc_moco.par'}
                 
    selectfiles = pe.Node(nio.SelectFiles(templates, 
                                        base_directory=data_dir),
                                        name='selectfiles')
                                        
    wf.connect(inputnode, 'subjid', selectfiles, 'subject')
    wf.connect(sessions_infosource, 'session', selectfiles, 'session') 
    wf.connect(sessions_infosource, 'session', ds, 'container')    

    ##########################################################################
    ########################    START   ######################################
    ##########################################################################
 

    ###########################################################################
    ########################    No. 1  ######################################
    #the script outputs only std DVARS
    DVARS = pe.Node(util.Function(input_names=['in_file', 'in_mask', 'out_std_name'],
                                  output_names=['out_std', 'out_nstd', 'out_vx_std'],
                                    function=compute_dvars),
                                    name='DVARS')
    
    DVARS.inputs.out_std_name = 'stdDVARS_pre.txt'                               
    wf.connect(selectfiles, 'prefiltered_detrend', DVARS, 'in_file')
    wf.connect(selectfiles, 'prefiltered_mask', DVARS, 'in_mask')
    wf.connect(DVARS, 'out_std', ds, 'QC.@DVARS')
    
    ###########################################################################
    ########################    No. 2   ######################################    
    # DEMAN and DETREND the data, which are used to get nuisance regressors
    
    def run_demean_detrend(in_file):
        import nibabel as nb
        import numpy as np
        import os
        from scipy.signal import detrend
        
        img = nb.load(in_file)
        imgseries = img.get_data().astype(np.float32)
        imgseries_new = detrend(imgseries, type='linear')
       
        new = nb.nifti1.Nifti1Image(imgseries_new, header=img.get_header(), affine=img.get_affine())
        out_file = os.path.join(os.getcwd(), 'prefiltered_func_data_demean_detrend.nii.gz')
        new.to_filename(out_file)
        del imgseries, imgseries_new, new
        return out_file

    demean_detrend = pe.Node(util.Function(input_names=['in_file'],
                                           output_names=['out_file'],
                                            function=run_demean_detrend),
                                            name='demean_detrend')
    
    
    
    wf.connect(selectfiles, 'prefiltered', demean_detrend, 'in_file')
    wf.connect(demean_detrend, 'out_file', ds, 'TEMP.@demean_detrend')
    ###########################################################################
    ########################    No. 3A   ######################################
    #PREPARE WM_CSF MASK    
    
    WM_CSF_msk = pe.Node(fsl.BinaryMaths(operation='add'),
                                         name='wm_csf_msk')
    
    wf.connect(selectfiles, 'WM_msk', WM_CSF_msk, 'in_file')
    wf.connect(selectfiles, 'CSF_msk', WM_CSF_msk, 'operand_file')
   
    #take the coverage of the masks from functional data (essentially multiply by the mask from functional data)
    func_msk = pe.Node(fsl.BinaryMaths(operation='mul',
                                          out_file='WM_CSFmsk.nii.gz'),  
                                          name='func_masking')
    
    wf.connect(WM_CSF_msk, 'out_file', func_msk, 'in_file')
    wf.connect(selectfiles, 'prefiltered_mask', func_msk, 'operand_file')
    wf.connect(func_msk, 'out_file', ds, 'TEMP.@masks')

    ###########################################################################
    ########################    No. 3B   ######################################
    #PREPARE MOTION REGRESSSORS FRISTON 24 AND TRENDS
    
    friston24 = pe.Node(util.Function(input_names=['in_file'],
                                      output_names=['out_file'],
                                      function=calc_friston_twenty_four),
                                      name='friston24')
    
    wf.connect(selectfiles, 'motion_par', friston24, 'in_file')
    wf.connect(friston24, 'out_file', ds, 'TEMP.@friston24')
    
    # linear and quadratic trends
    trends = pe.Node(util.Function(input_names=['nr_vols'],
                                   output_names=['out_file'],
                                    function=calc_trends),
                                      name='trends')
    
    def get_nr_vols(in_file):
        import nibabel as nb
        img = nb.load(in_file)
        return img.shape[3]
    
    wf.connect(demean_detrend, ('out_file', get_nr_vols), trends, 'nr_vols')
    wf.connect(trends, 'out_file', ds, 'TEMP.@trends')
    
    ###########################################################################
    ########################    No. 3C   ######################################
    #aCOMP_COR
    aCompCor = pe.Node(util.Function(input_names=['in_file', 'in_mask'],
                                  output_names=['out_file'],
                                  function=calc_compcor),
                                  name='aCompCor')
    
    wf.connect(demean_detrend, 'out_file', aCompCor, 'in_file')
    wf.connect(func_msk, 'out_file', aCompCor, 'in_mask')
    wf.connect(aCompCor, 'out_file', ds, 'TEMP.@aCompCor')
    
   
    ###########################################################################
    ########################    No. 4   ######################################
    #PREP the nuisance model
   
    #A is with Global Signal, and B is CompCor 
    def mergetxt(filelist, fname):
        import pandas as pd
        import os
        for n, f in enumerate(filelist):
            if n==0:
                data = pd.read_csv(f, header=None, sep='\t')
            else:
                data_new = pd.read_csv(f, header=None, sep='\t')
                data = pd.concat([data, data_new], axis=1)
        
        out_file = os.path.join(os.getcwd(), 'nuisance'+fname+'.mat')
        data.to_csv(out_file, index = False, header=None, sep='\t')
        return out_file
  
    merge_nuisance = pe.Node(util.Merge(3), infields=['in1', 'in2', 'in3'], name='merge_nuisance')
    
    wf.connect(aCompCor, 'out_file', merge_nuisance, 'in1')
    wf.connect(friston24, 'out_file', merge_nuisance, 'in2')
    wf.connect(trends, 'out_file', merge_nuisance, 'in3')

    nuisance_txt = pe.Node(util.Function(input_names=['filelist', 'fname'],
                              output_names=['out_file'],
                              function=mergetxt), 
                              name='nuisance_txt')
                              
    nuisance_txt.inputs.fname = '_model'    
    wf.connect(merge_nuisance, 'out', nuisance_txt, 'filelist')
    wf.connect(nuisance_txt, 'out_file', ds, 'TEMP.@nuisance_txt')

    ###########################################################################
    ########################    No. 5   ######################################
    #run nuisance regression on prefiltered raw data

    regression = pe.Node(fsl.GLM(demean=True), name='regression')
                                       
                                       
    regression.inputs.out_res_name = 'residuals.nii.gz'
    regression.inputs.out_f_name = 'residuals_fstats.nii.gz'
    regression.inputs.out_pf_name = 'residuals_pstats.nii.gz'
    regression.inputs.out_z_name = 'residuals_zstats.nii.gz'

    wf.connect(nuisance_txt, 'out_file', regression, 'design')
    wf.connect(selectfiles, 'prefiltered', regression, 'in_file')
    wf.connect(selectfiles, 'prefiltered_mask', regression, 'mask')
    
    wf.connect(regression, 'out_f', ds, 'REGRESSION.@out_f_name')
    wf.connect(regression, 'out_pf', ds, 'REGRESSION.@out_pf_name')
    wf.connect(regression, 'out_z', ds, 'REGRESSION.@out_z_name')
    
    ########################   FIX HEADER TR AFTER FSL_GLM   #################
    fixhd = pe.Node(fsl.utils.CopyGeom(), name='fixhd')
    
    wf.connect(regression, 'out_res', fixhd, 'dest_file')
    wf.connect(selectfiles, 'prefiltered', fixhd, 'in_file')
    wf.connect(fixhd, 'out_file', ds, 'REGRESSION.@res_out')

    ###########################################################################
    ########################    No. 6   ######################################
    #apply HP FILTER of 0.01Hz    
    #100/1.96/2 = 25.51
    hp_filter = pe.Node(fsl.maths.TemporalFilter(highpass_sigma=25.51,
                                                    out_file = 'residuals_hp01.nii.gz'),
                                                    name='highpass')
    
    wf.connect(fixhd, 'out_file', hp_filter, 'in_file')
    wf.connect(hp_filter, 'out_file', ds, 'TEMP.@hp')
    
    #add the mean back for smoothing
    addmean = pe.Node(fsl.BinaryMaths(operation='add',
                                         out_file='filtered_func_data_hp01.nii.gz'),
                                         name='addmean')
                                         
    wf.connect(hp_filter, 'out_file', addmean, 'in_file')
    wf.connect(selectfiles, 'prefiltered_detrend_Tmean', addmean, 'operand_file')   
    wf.connect(addmean, 'out_file', ds, '@out')

    ###########################################################################
    ########################    No. 7   ######################################
    ## COMPUTE POST DVARS    
    DVARSpost = pe.Node(util.Function(input_names=['in_file', 'in_mask', 'out_std_name'],
                                  output_names=['out_std', 'out_nstd', 'out_vx_std'],
                                    function=compute_dvars),
                                    name='DVARSpost')

    DVARSpost.inputs.out_std_name = 'stdDVARS_post.txt'                     
                  
    wf.connect(addmean, 'out_file', DVARSpost, 'in_file')
    wf.connect(selectfiles, 'prefiltered_mask', DVARSpost, 'in_mask')  
    wf.connect(DVARSpost, 'out_std', ds, 'QC.@DVARSpost')    

    
    ###########################################################################
    ########################    No. 8   ######################################
    #SMOOTHING of 6fwhm

    merge_datasets = pe.Node(util.Merge(2), infields=['in1', 'in2'], name='merge_datasets')
    

    wf.connect(addmean, 'out_file', merge_datasets, 'in1')
    wf.connect(selectfiles, 'prefiltered_detrend', merge_datasets, 'in2')
    
    median = pe.MapNode(fsl.utils.ImageStats(op_string='-k %s -p 50'), name='median', iterfield=['in_file'])

    wf.connect(merge_datasets, 'out', median, 'in_file')
    wf.connect(selectfiles, 'prefiltered_mask', median, 'mask_file')

    smooth = pe.MapNode(fsl.SUSAN(fwhm=6.0), name='smooth', iterfield=['in_file', 'brightness_threshold', 'usans', 'out_file'])
    smooth.inputs.out_file = ['filtered_func_data_hp01_sm6fwhm.nii.gz', 'prefiltered_func_data_detrend_sm6fwhm.nii.gz']
    
    merge_usans = pe.MapNode(util.Merge(2), infields=['in1', 'in2'], name='merge_usans', iterfield=['in2'])
    
    wf.connect(selectfiles, 'prefiltered_detrend_Tmean', merge_usans, 'in1')
    wf.connect(median, 'out_stat', merge_usans, 'in2')
    
    def getbtthresh(medianvals):
        return [0.75 * val for val in medianvals]
        
    def getusans(x):
        return [[tuple([val[0], 0.75 * val[1]])] for val in x]
        
    wf.connect(merge_datasets, 'out', smooth, 'in_file')
    wf.connect(median, ('out_stat', getbtthresh), smooth, 'brightness_threshold')
    wf.connect(merge_usans, ('out', getusans), smooth, 'usans')
    wf.connect(smooth, 'smoothed_file', ds, '@smoothout')
 
   
    ###########################################################################
    ########################    RUN   ######################################  
    wf.write_graph(dotfilename='wf.dot', graph2use='colored', format='pdf', simple_form=True)
    wf.run(plugin='MultiProc', plugin_args={'n_procs' : 2})
    #wf.run()
    return    














    