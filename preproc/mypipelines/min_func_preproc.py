# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:47:09 2016

@author: baczkowski
Based on https://github.com/juhuntenburg/pipelines/tree/master/src/lsd_lemon
"""
from __future__ import division
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.algorithms.misc as misc

def min_func_preproc(subject, 
                     sessions, 
                     data_dir, 
                     fs_dir,
                     wd,
                     sink,
                     TR,
                     EPI_resolution):
                         
    #initiate min func preproc workflow
    wf = pe.Workflow(name='MPP')
    wf.base_dir = wd
    wf.config['execution']['crashdump_dir'] = wf.base_dir + "/crash_files"
    
    ## set fsl output type to nii.gz
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    
    # I/O nodes
    inputnode = pe.Node(util.IdentityInterface(fields=['subjid',
                                                       'fs_dir']), 
                                                       name='inputnode')  
    inputnode.inputs.subjid = subject
    inputnode.inputs.fs_dir = fs_dir
    
    ds = pe.Node(nio.DataSink(base_directory=sink,
                              parameterization=False),
                              name='sink')    
                            
    ds.inputs.substitutions = [('moco.nii.gz.par', 'moco.par'),
                               ('moco.nii.gz_', 'moco_')]
    
    #infosource to interate over sessions: COND, EXT1, EXT2
    sessions_infosource = pe.Node(util.IdentityInterface(fields=['session']),name='session')
    sessions_infosource.iterables = [('session', sessions)]  
    
    #select files
    templates = {'func_data': '{session}/func_data.nii.gz',
                 'T1_brain': 'T1/T1_brain.nii.gz',
                 'wmedge': 'T1/MASKS/aparc_aseg.WMedge.nii.gz'}
                 
    selectfiles = pe.Node(nio.SelectFiles(templates, 
                                        base_directory=data_dir),
                                        name='selectfiles')
                                        
    
    wf.connect(sessions_infosource, 'session', selectfiles, 'session') 
    wf.connect(sessions_infosource, 'session', ds, 'container')    
     
    ##########################################################################
    ########################    START   ######################################
    ##########################################################################
    
    ###########################################################################
    ########################    No. 3   ######################################
    
    #change the data type to float
    fsl_float = pe.Node(fsl.maths.MathsCommand(output_datatype='float'), 
                                        name='fsl_float')
                                            

    wf.connect(selectfiles, 'func_data', fsl_float, 'in_file')
    
    ###########################################################################
    ########################    No. 4   ######################################
    
    #get FD from fsl_motion_outliers
    FD = pe.Node(fsl.MotionOutliers(out_file='func_data_FD_outliers.txt',
                                    out_metric_values='func_data_FD.txt',
                                    metric='fd'), 
                                    name='FD')
                                       
    wf.connect(fsl_float, 'out_file', FD, 'in_file')
    wf.connect(FD, 'out_metric_values', ds, 'QC.@FD')
    wf.connect(FD, 'out_file', ds, 'QC.@FDoutliers')

    ###########################################################################
    ########################    No. 5   ######################################

    #slice timing correction: sequential ascending
    slicetimer = pe.Node(fsl.SliceTimer(index_dir=False,
                                        interleaved=False,
                                        #slice_direction=3, #z direction
                                        time_repetition=TR,
                                        out_file='func_data_stc.nii.gz'), 
                                        name='slicetimer')
    
    wf.connect(fsl_float, 'out_file', slicetimer, 'in_file')
    wf.connect(slicetimer, 'slice_time_corrected_file', ds, 'TEMP.@slicetimer')

    ###########################################################################
    ########################    No. 6   ######################################
    #do realignment to the middle or first volume
    mcflirt = pe.Node(fsl.MCFLIRT(save_mats=True,
                               save_plots=True,
                               save_rms=True,
                               ref_vol=1,
                               out_file='func_data_stc_moco.nii.gz'),
                               name='mcflirt')
    
    wf.connect(slicetimer, 'slice_time_corrected_file', mcflirt, 'in_file')
    wf.connect(mcflirt, 'out_file', ds, 'TEMP.@mcflirt')
    wf.connect(mcflirt, 'par_file', ds, 'MOCO.@par_file')
    wf.connect(mcflirt, 'rms_files', ds, 'MOCO.@rms_files')
    wf.connect(mcflirt, 'mat_file', ds, 'MOCO_MAT.@mcflirt')

    # plot motion parameters
    rotplotter = pe.Node(fsl.PlotMotionParams(in_source='fsl',
                                              plot_type='rotations',
                                              out_file='rotation.png'),
                                              name='rotplotter')
    
    
    transplotter = pe.Node(fsl.PlotMotionParams(in_source='fsl',
                                                plot_type='translations',
                                                out_file='translation.png'),
                                                name='transplotter')

    dispplotter = pe.Node(interface=fsl.PlotMotionParams(in_source='fsl',
                                                         plot_type='displacement',
                                                         out_file='displacement.png'),
                                                         name='dispplotter')

    
    wf.connect(mcflirt, 'par_file', rotplotter, 'in_file') 
    wf.connect(mcflirt, 'par_file', transplotter, 'in_file')
    wf.connect(mcflirt, 'rms_files', dispplotter, 'in_file')    
    wf.connect(rotplotter, 'out_file', ds, 'PLOTS.@rotplot')
    wf.connect(transplotter, 'out_file', ds, 'PLOTS.@transplot')
    wf.connect(dispplotter, 'out_file', ds, 'PLOTS.@disppplot')
    
    #calculate tSNR and the mean
 
    moco_Tmean = pe.Node(fsl.maths.MathsCommand(args='-Tmean',
                                                out_file='moco_Tmean.nii.gz'),
                                                name='moco_Tmean') 
                                                
    moco_Tstd = pe.Node(fsl.maths.MathsCommand(args='-Tstd',
                                                out_file='moco_Tstd.nii.gz'),
                                                name='moco_Tstd')
                                                
    tSNR0 = pe.Node(fsl.maths.MultiImageMaths(op_string='-div %s',
                                               out_file='moco_tSNR.nii.gz'),
                                               name='moco_tSNR')
                                            
    wf.connect(mcflirt, 'out_file', moco_Tmean, 'in_file')
    wf.connect(mcflirt, 'out_file', moco_Tstd, 'in_file')
    wf.connect(moco_Tmean, 'out_file', tSNR0, 'in_file')
    wf.connect(moco_Tstd, 'out_file', tSNR0, 'operand_files')
    wf.connect(moco_Tmean, 'out_file', ds, 'TEMP.@moco_Tmean')
    wf.connect(moco_Tstd, 'out_file', ds, 'TEMP.@moco_Tstd')
    wf.connect(tSNR0, 'out_file', ds, 'TEMP.@moco_Tsnr')
    
    ###########################################################################
    ########################    No. 7   ######################################
    
    #bias field correction of mean epi for better coregistration
    bias = pe.Node(fsl.FAST(img_type=2,
                            #restored_image='epi_Tmeanrestored.nii.gz',
                            output_biascorrected=True,
                            out_basename='moco_Tmean',
                            no_pve=True,
                            probability_maps=False), 
                            name='bias')
                            
                            
    wf.connect(moco_Tmean, 'out_file', bias, 'in_files')
    wf.connect(bias, 'restored_image', ds, 'TEMP.@restored_image')
    
    #co-registration to anat using FS BBregister and mean EPI
    bbregister = pe.Node(fs.BBRegister(subject_id=subject,
                                       subjects_dir=fs_dir,
                                       contrast_type='t2',
                                       init='fsl',
                                       out_fsl_file='func2anat.mat',
                                       out_reg_file='func2anat.dat',
                                       registered_file='moco_Tmean_restored2anat.nii.gz',
                                       epi_mask=True),
                                       name='bbregister')
                            
    wf.connect(bias, 'restored_image', bbregister, 'source_file') 
    wf.connect(bbregister, 'registered_file', ds, 'TEMP.@registered_file')
    wf.connect(bbregister, 'out_fsl_file', ds, 'COREG.@out_fsl_file')
    wf.connect(bbregister, 'out_reg_file', ds, 'COREG.@out_reg_file')
    wf.connect(bbregister, 'min_cost_file', ds, 'COREG.@min_cost_file')

    #inverse func2anat mat
    inverseXFM = pe.Node(fsl.ConvertXFM(invert_xfm=True,
                                        out_file='anat2func.mat'),
                                        name='inverseXFM')
                                        
    wf.connect(bbregister, 'out_fsl_file', inverseXFM, 'in_file')
    wf.connect(inverseXFM, 'out_file', ds, 'COREG.@out_fsl_file_inv')
    
    #plot the corregistration quality
    slicer = pe.Node(fsl.Slicer(middle_slices=True,
                                out_file='func2anat.png'),
                                name='slicer')

    wf.connect(selectfiles, 'wmedge', slicer, 'image_edges')                 
    wf.connect(bbregister, 'registered_file', slicer, 'in_file')
    wf.connect(slicer, 'out_file', ds, 'PLOTS.@func2anat')
    
    ###########################################################################
    ########################    No. 8   ######################################
    #MOCO and COREGISTRATION
    
    #resample T1 to EPI resolution to use it as a reference image
    resample_T1 = pe.Node(fsl.FLIRT(datatype='float',
                                 apply_isoxfm=EPI_resolution,
                                 out_file='T1_brain_EPI.nii.gz'),
                                 #interp='nearestneighbour'),keep spline so it looks nicer
                                 name='resample_T1')
    
    wf.connect(selectfiles, 'T1_brain', resample_T1, 'in_file')
    wf.connect(selectfiles, 'T1_brain', resample_T1, 'reference')
    wf.connect(resample_T1, 'out_file', ds, 'COREG.@resample_T1')
    
    #concate matrices (moco and func2anat) volume-wise        
    concat_xfm = pe.MapNode(fsl.ConvertXFM(concat_xfm=True),
                                       iterfield=['in_file'],
                                        name='concat_xfm')
    
    wf.connect(mcflirt, 'mat_file', concat_xfm, 'in_file')    
    wf.connect(bbregister, 'out_fsl_file', concat_xfm, 'in_file2')
    wf.connect(concat_xfm, 'out_file', ds, 'MOCO2ANAT_MAT.@concat_out')
    

    #split func_data
    split = pe.Node(fsl.Split(dimension='t'), name='split')
    
    wf.connect(slicetimer, 'slice_time_corrected_file', split, 'in_file')
    
    #motion correction and corregistration in one interpolation step
    flirt = pe.MapNode(fsl.FLIRT(apply_xfm=True,
                                 interp='spline',
                                 datatype='float'),
                                 iterfield=['in_file', 'in_matrix_file'],
                                    name='flirt')
                                    
    wf.connect(split, 'out_files', flirt, 'in_file')
    wf.connect(resample_T1, 'out_file', flirt, 'reference')
    wf.connect(concat_xfm, 'out_file', flirt, 'in_matrix_file')
    
    #merge the files to have 4d dataset motion corrected and co-registerd to T1
    merge = pe.Node(fsl.Merge(dimension='t',
                              merged_file='func_data_stc_moco2anat.nii.gz'),
                              name='merge')
                              
    wf.connect(flirt, 'out_file', merge, 'in_files')
    wf.connect(merge, 'merged_file', ds, 'TEMP.@merged')
            

    ###########################################################################
    ########################    No. 9   ######################################
    
    #run BET on co-registered EPI in 1mm and get the mask
    bet = pe.Node(fsl.BET(mask=True,
                          functional=True,
                          out_file='moco_Tmean_restored2anat_BET.nii.gz'),
                          name='bet')

    wf.connect(bbregister, 'registered_file', bet, 'in_file')
    wf.connect(bet, 'out_file', ds, 'TEMP.@func_data_example')
    wf.connect(bet, 'mask_file', ds, 'TEMP.@func_data_mask')
    
    #resample BET mask to EPI resolution
    resample_mask = pe.Node(fsl.FLIRT(datatype='int',
                                      apply_isoxfm=EPI_resolution,
                                      interp='nearestneighbour',
                                      out_file='prefiltered_func_data_mask.nii.gz'),
                                      name='resample_mask')
    
    wf.connect(bet, 'mask_file', resample_mask, 'in_file')
    wf.connect(resample_T1, 'out_file', resample_mask, 'reference')
    wf.connect(resample_mask, 'out_file', ds, '@mask')

    #apply the mask to 4D data to get rid of the "eyes and the rest"
    mask4D = pe.Node(fsl.maths.ApplyMask(),name='mask')
                                       
    wf.connect(merge, 'merged_file', mask4D, 'in_file')
    wf.connect(resample_mask, 'out_file', mask4D, 'mask_file')

    ###########################################################################
    ########################    No. 10   ######################################

    #get the values necessary for intensity normalization
    median = pe.Node(fsl.utils.ImageStats(op_string='-k %s -p 50'), name='median')
    
    wf.connect(resample_mask, 'out_file', median, 'mask_file')
    wf.connect(mask4D, 'out_file', median, 'in_file')
    
    #compute the scaling factor
    def get_factor(val):

        factor=10000/val
        return factor
    
    get_scaling_factor = pe.Node(util.Function(input_names=['val'],
                                               output_names=['out_val'],
                                                function=get_factor), 
                                                name='scaling_factor')
                                                
    #normalize the 4D func data with one scaling factor                                             
    multiplication = pe.Node(fsl.maths.BinaryMaths(operation='mul',
                                                   out_file='prefiltered_func_data.nii.gz'),
                                                   name='multiplication')
    


    wf.connect(median, 'out_stat', get_scaling_factor, 'val')    
    wf.connect(get_scaling_factor, 'out_val', multiplication, 'operand_value')
    wf.connect(mask4D, 'out_file', multiplication, 'in_file')
    wf.connect(multiplication, 'out_file', ds, '@prefiltered_func_data')
    
    ###########################################################################
    ########################    No. 11   ######################################  
    
    #calculate tSNR and the mean of the new prefiltered and detrend dataset   
    
    tsnr_detrend = pe.Node(misc.TSNR(regress_poly=1,
                             detrended_file='prefiltered_func_data_detrend.nii.gz',
                             mean_file='prefiltered_func_data_detrend_Tmean.nii.gz',
                             tsnr_file='prefiltered_func_data_detrend_tSNR.nii.gz'),
                             name='tsnr_detrend')  
    
    wf.connect(multiplication, 'out_file', tsnr_detrend, 'in_file')
    wf.connect(tsnr_detrend, 'tsnr_file', ds, 'QC.@tsnr_detrend')
    wf.connect(tsnr_detrend, 'mean_file', ds, 'QC.@detrend_mean_file') 
    wf.connect(tsnr_detrend, 'detrended_file', ds, '@detrend_file') 
                           
    
    #resample the EPI mask to original EPI dimensions
    convert2func = pe.Node(fsl.FLIRT(apply_xfm=True,
                                        interp='nearestneighbour',
                                        out_file='func_data_mask2func.nii.gz'), 
                                        name='conver2func')
                                        
    wf.connect(resample_mask, 'out_file', convert2func, 'in_file')
    wf.connect(bias, 'restored_image', convert2func, 'reference')
    wf.connect(inverseXFM, 'out_file', convert2func, 'in_matrix_file')
    wf.connect(convert2func, 'out_file', ds, 'QC.@inv')
    
   
    ###########################################################################
    ########################    RUN   ######################################  
    wf.write_graph(dotfilename='wf.dot', graph2use='colored', format='pdf', simple_form=True)
    wf.run(plugin='MultiProc', plugin_args={'n_procs' : 2})
    #wf.run()
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    