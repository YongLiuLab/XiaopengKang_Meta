""" utility module of Mask

Class:
    Mask(object): utility class to help manipulate array

Author: Kang Xiaopeng
Data: 2020/03/05
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
import nibabel as nib
import numpy as np

def nomalize(data):
    """nomalize data using global min,max
    """
    _min, _max = data.min(), data.max()
    return (data-_min)/(_max-_min)

class Mask(object):
    """docstring for Mask

    Attributes:
        file_dir: file dir that this mask are in
        filename: mask's filename
        data: ndarray

    Function:
        get_mask_data(label): get this mask's array data of label
        get_shape(): get data shape
        get_nonzero_index(): get index of all nozero element
        get_masked_data(array, label): get masked array of label
        get_masked_volume(array, label): get sum of masked array of label
        get_labels(): get all label except 0
        get_all_masked_volume(): get all label's masked volume
        get_masked_count(): get lebel's count
        get_masked_mean(): get mean value of label region
        get_all_masked_mean(): get all mean value of label
    """
    def __init__(self, data):
        self.data = data

    def get_mask_data(self, label=None):
        """get this mask's array data

        Returns:
            ndarray
        """
        if label is None:
            mask_data = nomalize(self.data)
        else:
            # True and False in mask_data will be treated as 1 and 0
            mask_data = self.data==label
        return mask_data

    def get_shape(self):
        return self.data.shape

    def get_nonzero_index(self):
        return np.transpose(np.nonzero(self.data))

    def get_masked_data(self, array, label=None):
        """return masked array
        Args:
            array: np.ndarray
            label: mask label
        Return:
            masked array
        """
        mask_data = self.get_mask_data(label)
        return np.multiply(mask_data, array)

    def get_masked_volume(self, array, label):
        return np.sum(self.get_masked_data(array, label))

    def get_labels(self):
        """return all unique label, exclude 0 cause it usually doesn't count as label. 
        Return:
            1D ndarray of unique labels without 0
        """
        unique = np.unique(self.data)
        return unique[unique != 0]

    def get_all_masked_volume(self, array):
        volumes = {}
        labels = self.get_labels()
        for i in labels:
            volumes[i] = self.get_masked_volume(array, i)
        return volumes
    
    def get_masked_count(self, label):
        mask_data = self.get_mask_data(label)
        count = np.sum(mask_data==True)
        return count

    def get_masked_mean(self, array, label):
        volume = self.get_masked_volume(array, label)
        count = self.get_masked_count(label)
        return volume / count

    def get_all_masked_mean(self, array):
        means = {}
        labels = self.get_labels()
        for i in labels:
            means[i] = self.get_masked_mean(array, i)
        return means


class NiiMask(Mask):
    def __init__(self, filepath):
        self.filepath = filepath
        self.nii = nib.load(filepath)
        filename = os.path.split(filepath)[1]
        super().__init__(filename, np.asarray(self.nii.dataobj))

    def load_nii(self, nii):
        array = np.asarray(nii.dataobj)
        array = np.nan_to_num(array)
        array = np.reshape(array, newshape=self.shape)
        return array

    def get_all_masked_volume(self, nii):
        array = self.load_nii(nii)
        return super().get_all_masked_volume(array)

    def get_all_masked_mean(self, nii):
        array = self.load_nii(nii)
        return super().get_all_masked_mean(array)

    def get_all_statistics(self, nii):
        array = self.load_nii(nii)
        return super().get_all_statistics(array)

    def save_values(self, values, out_path, dtype=np.float32):
        data = self.data.astype(dtype)
        ll = self.labels.tolist()

        for k, v in values.items():
            if k in ll:
                data[data==k] = v
                ll.remove(k)
        for i in ll:
            data[data==i] = 0
        nii = utils.gen_nii(data, self.nii, out_path, dtype)
        return nii

class GiiMask(Mask):
    def __init__(self, filepath):
        self.filepath = filepath
        self.gii = nib.load(filepath)
        array = self.gii.darrays[0].data
        filename = os.path.split(filepath)[1]
        super().__init__(filename, array)

    def load_gii(self, gii):
        array = gii.darrays[0].data
        array = np.nan_to_num(array)
        array = np.reshape(array, newshape=self.shape)
        return array

    def get_all_masked_volume(self, gii):
        array = self.load_gii(gii)
        return super().get_all_masked_volume(array)

    def get_all_masked_mean(self, gii):
        array = self.load_gii(gii)
        return super().get_all_masked_mean(array)

    def get_all_statistics(self, gii):
        array = self.load_gii(gii)
        return super().get_all_statistics(array)