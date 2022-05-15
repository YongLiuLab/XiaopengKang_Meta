""" Some helper function.

Function:
    load_array(path): load nii's array.
    load_arrays(pathes, axis): load niis' array then stack them along 'axis'.
    cal_mean_std_n(arrays, axis): calculate arrays' mean, std, count along 'axis'.
    gen_nii(array, template_nii, path): generate nii file using template's header and affine

Author: Kang Xiaopeng
Data: 2020/03/06
E-mail: kangxiaopeng2018@ia.ac.cn

This file is part of meta_analysis.

meta_analysis is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

meta_analysis is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with meta_analysis.  If not, see <https://www.gnu.org/licenses/>.
"""
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