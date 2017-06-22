# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 19:07:27 2016

@author: baczkowski
"""
#from franz

def extract_signal_from_tissue(in_file, mask_file):
#    import numpy as np
    import nibabel as nb
    
    def safe_shape(*vol_data):
        """
        Checks if the volume (first three dimensions) of multiple ndarrays
        are the same shape.
        Parameters
        ----------
        vol_data0, vol_data1, ..., vol_datan : ndarray
            Volumes to check
        Returns
        -------
        same_volume : bool
            True only if all volumes have the same shape.
        """
        same_volume = True

        first_vol_shape = vol_data[0].shape[:3]
        for vol in vol_data[1:]:
            same_volume &= (first_vol_shape == vol.shape[:3])

        return same_volume 
    
    
    try:
        data = nb.load(in_file).get_data().astype('float64')
    except:
        raise MemoryError('Unable to load %s' % in_file)

    try:
        mask = nb.load(mask_file).get_data().astype('bool')
    except:
        raise MemoryError('Unable to load %s' % mask)

    if not safe_shape(data, mask):
        raise ValueError('Spatial dimensions for data and mask %s do not match' % mask)

    tissue_sigs = data[mask]
#    file_sigs = os.path.join(os.getcwd(), 'signals.npy')
#    np.save(file_sigs, tissue_sigs)
#    del tissue_sigs

    return tissue_sigs

def calc_mean_tissue_sigs(tissue_sigs):
    import numpy as np
    mean_tissue_sigs = np.mean(tissue_sigs, axis=0)
    return mean_tissue_sigs
    
def save_tissue_sigs(tissue_sigs, filename):
    
    def _gen_fname(in_file, suffix, ext=None):
        import os.path as op
        fname, in_ext = op.splitext(op.basename(in_file))

        if in_ext == '.gz':
            fname, in_ext2 = op.splitext(fname)
            in_ext = in_ext2 + in_ext

        if ext is None:
            ext = in_ext

        if ext.startswith('.'):
            ext = ext[1:]

        return op.abspath('{}_{}.{}'.format(fname, suffix, ext))
    
    
    import numpy as np
    
    out_file = _gen_fname(filename, 'tissue_signals' ,'.txt')
    np.savetxt(out_file, tissue_sigs, fmt='%0.8f', delimiter=' ')
    return out_file   
    
    
    
    
    
    
    
    