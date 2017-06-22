# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:25:32 2016

@author: baczkowski
"""

import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
from nipype import config


def run_denoise(subject): 
    
    import os
    from mypipelines.denoise_acompcor import denoise

    
    dynorphin_dir = '/nobackup/usbekistan4/baczkowski/DYNORPHIN'
    data_dir = os.path.join(dynorphin_dir, 'sink_dir')
    wd = os.path.join(dynorphin_dir, 'working_dir/DENOISE_aCompCor/', subject)
    sink = os.path.join(dynorphin_dir, 'sink_dir/DENOISE_aCompCor/', subject)

    TR=1.9600
    sessions=['EXT1', 'EXT2']
    
    denoise(subject,
            sessions,
            data_dir, 
            wd,
            sink,
            TR)
    
    return

subjects = [#'02']
            '01']
#            '01','02', '03', '04', '05', '06', '07', '08', '09',
#            '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
#            '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
#            '30', '31', '32', '33', '34']



wd = '/nobackup/usbekistan4/baczkowski/DYNORPHIN/working_dir/'


wf = pe.Workflow(name='DENOISE_aCompCor')
wf.base_dir = wd
wf.config['execution']['crashdump_dir'] = wf.base_dir + "/crash_files"

nipype_cfg = dict(logging=dict(workflow_level='DEBUG'), execution={'stop_on_first_crash': False,
                                                                   'remove_unnecessary_outputs': False,
                                                                   'job_finished_timeout': 120})
config.update_config(nipype_cfg)


infosource_sub = pe.Node(util.IdentityInterface(fields=['subject', 'sessions']), name='infosource_sub')
infosource_sub.iterables = [('subject', subjects)]



run_func = pe.Node(util.Function(input_names=['subject'],#, 'sessions'],
                                   output_names=[], #'filelist'
                                   function=run_denoise), 
                                   name='run_func')
                                   
wf.connect(infosource_sub, 'subject', run_func, 'subject')

wf.write_graph(dotfilename='wf.dot', graph2use='colored', format='pdf', simple_form=True)
#wf.run(plugin='CondorDAGMan')
wf.run(plugin='MultiProc', plugin_args={'n_procs': 10})