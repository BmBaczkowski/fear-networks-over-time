# -*- coding: utf-8 -*-

#FROM C-PAC

def calc_friston_twenty_four(in_file):
    """
    Method to calculate friston twenty four parameters
    
    Parameters
    ----------
    in_file: string
        input movement parameters file from motion correction
    
    Returns
    -------
    new_file: string
        output 1D file containing 24 parameter values
        
    """

    import numpy as np
    import os

    new_data = None

    data = np.genfromtxt(in_file)

    data_squared = data ** 2

    new_data = np.concatenate((data, data_squared), axis=1)

    data_roll = np.roll(data, 1, axis=0)

    data_roll[0] = 0

    new_data = np.concatenate((new_data, data_roll), axis=1)

    data_roll_squared = data_roll ** 2

    new_data = np.concatenate((new_data, data_roll_squared), axis=1)

#    def _gen_fname(in_file, suffix, ext=None):
#        import os.path as op
#        fname, in_ext = op.splitext(op.basename(in_file))
#
#        if in_ext == '.gz':
#            fname, in_ext2 = op.splitext(fname)
#            in_ext = in_ext2 + in_ext
#
#        if ext is None:
#            ext = in_ext
#
#        if ext.startswith('.'):
#            ext = ext[1:]
#
#        return op.abspath('{}_{}.{}'.format(fname, suffix, ext))

    #out_file = _gen_fname(in_file, 'friston24' ,'.txt')
    out_file = os.path.join(os.getcwd(), 'friston24.txt')
    np.savetxt(out_file, new_data, fmt='%0.8f', delimiter='\t')
    del new_data
    return out_file 

   
