##
# Code taken from Nipype 0.13.0 in development and adopted
##
from __future__ import print_function, division, unicode_literals, absolute_import


def compute_dvars(in_file, in_mask, out_std_name=False, out_std=True, out_nstd=False, out_vx_std=False, remove_zerovariance=True):
    """
    Compute the :abbr:`DVARS (D referring to temporal
    derivative of timecourses, VARS referring to RMS variance over voxels)`
    [Power2012]_.

    Particularly, the *standardized* :abbr:`DVARS (D referring to temporal
    derivative of timecourses, VARS referring to RMS variance over voxels)`
    [Nichols2013]_ are computed.

    .. [Nichols2013] Nichols T, `Notes on creating a standardized version of
         DVARS <http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-\
         research/nichols/scripts/fsl/standardizeddvars.pdf>`_, 2013.

    .. note:: Implementation details

      Uses the implementation of the `Yule-Walker equations
      from nitime
      <http://nipy.org/nitime/api/generated/nitime.algorithms.autoregressive.html\
      #nitime.algorithms.autoregressive.AR_est_YW>`_
      for the :abbr:`AR (auto-regressive)` filtering of the fMRI signal.

    :param numpy.ndarray func: functional data, after head-motion-correction.
    :param numpy.ndarray mask: a 3D mask of the brain
    :param bool output_all: write out all dvars
    :param str out_file: a path to which the standardized dvars should be saved.
    :return: the standardized DVARS

    """
    import numpy as np
    import nibabel as nb
    from nitime.algorithms import AR_est_YW
    import warnings

##############################################################################
#########                   HELPER FUNCTIONS                    ##############
##############################################################################

    def regress_poly(degree, data, remove_mean=False, axis=-1):
        ''' returns data with degree polynomial regressed out.
        Be default it is calculated along the last axis (usu. time).
        If remove_mean is True (default), the data is demeaned (i.e. degree 0).
        If remove_mean is false, the data is not.
        '''
        from numpy.polynomial import Legendre
        datashape = data.shape
        timepoints = datashape[axis]
    
        # Rearrange all voxel-wise time-series in rows
        data = data.reshape((-1, timepoints))
    
        # Generate design matrix
        X = np.ones((timepoints, 1)) # quick way to calc degree 0
        for i in range(degree):
            polynomial_func = Legendre.basis(i + 1)
            value_array = np.linspace(-1, 1, timepoints)
            X = np.hstack((X, polynomial_func(value_array)[:, np.newaxis]))
    
        # Calculate coefficients
        betas = np.linalg.pinv(X).dot(data.T)
    
        # Estimation
        if remove_mean:
            datahat = X.dot(betas).T
        else: # disregard the first layer of X, which is degree 0
            datahat = X[:, 1:].dot(betas[1:, ...]).T
        regressed_data = data - datahat
    
        # Back to original shape
        return regressed_data.reshape(datashape)        
    
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
        
##############################################################################
#########                           START                       ##############
##############################################################################        
    func = nb.load(in_file).get_data().astype(np.float32)
    mask = nb.load(in_mask).get_data().astype(np.uint8)

    if len(func.shape) != 4:
        raise RuntimeError(
            "Input fMRI dataset should be 4-dimensional")

    # Robust standard deviation
    func_sd = (np.percentile(func, 75, axis=3) -
               np.percentile(func, 25, axis=3)) / 1.349
    func_sd[mask <= 0] = 0

    if remove_zerovariance:
        # Remove zero-variance voxels across time axis 

        new_mask = mask.copy()
        new_mask[func_sd == 0] = 0
        mask = new_mask
        

    idx = np.where(mask > 0)
    mfunc = func[idx[0], idx[1], idx[2], :]

    # Demean
    mfunc = regress_poly(0, mfunc, remove_mean=True).astype(np.float32)

    # Compute (non-robust) estimate of lag-1 autocorrelation
    ar1 = np.apply_along_axis(AR_est_YW, 1, mfunc, 1)[:, 0]

    # Compute (predicted) standard deviation of temporal difference time series
    diff_sdhat = np.squeeze(np.sqrt(((1 - ar1) * 2).tolist())) * func_sd[mask > 0].reshape(-1)
    diff_sd_mean = diff_sdhat.mean()

    # Compute temporal difference time series
    func_diff = np.diff(mfunc, axis=1)

    # DVARS (no standardization)
    dvars_nstd = func_diff.std(axis=0)

    # standardization
    dvars_stdz = dvars_nstd / diff_sd_mean

    with warnings.catch_warnings(): # catch, e.g., divide by zero errors
        warnings.filterwarnings('error')

        # voxelwise standardization
        diff_vx_stdz = func_diff / np.array([diff_sdhat] * func_diff.shape[-1]).T            
        dvars_vx_stdz = diff_vx_stdz.std(axis=0, ddof=1)

    
    import os
    if out_std:
        if out_std_name:
            out_std = os.path.join(os.getcwd(), out_std_name)
        else:
            out_std = os.path.join(os.getcwd(), 'stdDVARS.txt')
        np.savetxt(out_std, dvars_stdz, fmt=b'%0.6f')
    if out_nstd:
        out_nstd = os.path.join(os.getcwd(), 'nstdDVARS.txt')
        np.savetxt(out_nstd, dvars_nstd, fmt=b'%0.6f')
    if out_vx_std:
        out_vx_std = os.path.join(os.getcwd(), 'vxstdDVARS.txt')
        np.savetxt(out_vx_std, dvars_vx_stdz, fmt=b'%0.6f')

    del dvars_stdz, dvars_nstd, dvars_vx_stdz, func, mask
    
    return  out_std, out_nstd, out_vx_std

