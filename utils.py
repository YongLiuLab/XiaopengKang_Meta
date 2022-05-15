import os
import numpy as np
import nibabel as nib

def load_array(path, dtype=np.float32):
    nii = nib.load(path)
    array = np.asarray(nii.dataobj, dtype=dtype)
    array = np.nan_to_num(array)
    return array

def load_arrays(pathes, dtype=np.float32, axis=0):
    arrays = np.array([])
    if pathes:
        arrays = np.stack([load_array(path, dtype) for path in pathes], axis=axis)
    return arrays

def cal_mean_std_n(arrays, axis=0):
    arrays = np.asarray(arrays)
    mean = np.mean(arrays, axis=axis)
    std = np.std(arrays, axis=axis)
    n = arrays.shape[axis]
    return mean, std, n

def gen_nii(array, template_nii, path=None, dtype=np.float32):
    """ generate nii file using template's header and affine
        if input path then save nii in disk.
    Args:
        array: array to form nii
        template_nii: nibabel Nifti1 instance, use its affine and header
        path: string, path to store nii file
    Return:
        nii: nibabel Nifti1 instance 
    """
    affine = template_nii.affine
    header = template_nii.header
    header.set_data_dtype(dtype)
    nii = nib.Nifti1Image(array, affine, header)
    if path:
        filename, extension = os.path.splitext(path)
        if not extension or extension != '.nii':
            extension = '.nii'
        path = filename + extension
        nib.nifti1.save(nii, path)
    return nii