# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 21:39:26 2016

@author: baczkowski
"""
#taken from Nipype 0.13.0 in development

def calc_compcor(in_file, in_mask, num_components=5, use_regress_poly=False, regress_poly_degree=1):
    import nibabel as nb
    import numpy as np
    import scipy.linalg as linalg
    import os
    
    def regress_poly(degree, data, remove_mean=True, axis=-1):
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
    
    def _compute_tSTD(M, x):
        stdM = np.std(M, axis=0)
        # set bad values to x
        stdM[stdM == 0] = x
        stdM[np.isnan(stdM)] = x
        return stdM
    
    imgseries = nb.load(in_file).get_data().astype(np.float32)
    mask = nb.load(in_mask).get_data().astype(np.uint8)

    if imgseries.shape[:3] != mask.shape:
        raise ValueError('Inputs for CompCor, func {} and mask {}, do not have matching '
                         'spatial dimensions ({} and {}, respectively)'
                         .format(in_file, in_mask,
                                 imgseries.shape[:3], mask.shape))

    voxel_timecourses = imgseries[mask > 0]
    # Zero-out any bad values
    voxel_timecourses[np.isnan(np.sum(voxel_timecourses, axis=1)), :] = 0

    # from paper:
    # "The constant and linear trends of the columns in the matrix M were
    # removed [prior to ...]"
    degree = regress_poly_degree if use_regress_poly else 0
    voxel_timecourses = regress_poly(degree, voxel_timecourses)

    # "Voxel time series from the noise ROI (either anatomical or tSTD) were
    # placed in a matrix M of size Nxm, with time along the row dimension
    # and voxels along the column dimension."
    M = voxel_timecourses.T
 
    # "[... were removed] prior to column-wise variance normalization."
    M = M / _compute_tSTD(M, 1)

    # "The covariance matrix C = MMT was constructed and decomposed into its
    # principal components using a singular value decomposition."
    u, _, _ = linalg.svd(M, full_matrices=False)
    components = u[:, :num_components]

    out_file = os.path.join(os.getcwd(), 'compCor_components.txt')
    np.savetxt(out_file, components, fmt=b"%.10f", delimiter='\t')
    
    return out_file

