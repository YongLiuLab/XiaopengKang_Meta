"""
TODO:add license
"""

import os
import nibabel as nib
import numpy as np

import utils

def nomalize(data):
    """nomalize data using global min,max
    """
    _min, _max = data.min(), data.max()
    return (data-_min)/(_max-_min)

class Mask(object):
    def __init__(self, filename, data):
        self.filename = filename
        self.data = np.nan_to_num(data).astype(np.int16)
        self.shape = data.shape
        labels = np.unique(self.data)
        # Remove 0 as it usually don't act as label
        self.labels = labels[labels!=0]
        self.indices = np.transpose(np.nonzero(self.data))

    def get_label_bool(self, label):
        """get bool array by label
        Args:
            label: int, label in self.labels
        Returns:
            ndarray: bool array of label
        """
        assert label in self.labels
        # Bool element will be treated as 1 and 0 for further calculation
        return self.data==label

    def get_masked_data(self, array, label):
        """return masked array
        Args:
            array: ndarray
            label: mask label
        Return:
            masked array
        """
        bool_array = self.get_label_bool(label)
        return np.multiply(array, bool_array)

    def get_masked_volume(self, array, label):
        return np.sum(self.get_masked_data(array, label))

    def get_masked_mean(self, array, label):
        # get mean exclude 0
        summed = np.sum(self.get_masked_data(array, label))
        n = np.count_nonzero(self.get_masked_data(array, label))
        return summed/n

    def get_all_masked_volume(self, array):
        volumes = {}
        labels = self.labels
        for i in labels:
            volumes[i] = self.get_masked_volume(array, i)
        return volumes

    def get_all_masked_mean(self, array):
        means = {}
        labels = self.labels
        for i in labels:
            means[i] = self.get_masked_mean(array, i)
        return means

class NiiMask(Mask):
    def __init__(self, filepath):
        self.filepath = filepath
        self.nii = nib.load(filepath)
        filename = os.path.split(filepath)[1]
        super().__init__(filename, np.asarray(self.nii.dataobj))

    def get_all_masked_volume(self, nii):
        array = np.asarray(nii.dataobj)
        array = np.nan_to_num(array)
        array = np.reshape(array, newshape=self.shape)
        return super().get_all_masked_volume(array)

    def get_all_masked_mean(self, nii):
        array = np.asarray(nii.dataobj)
        array = np.nan_to_num(array)
        array = np.reshape(array, newshape=self.shape)
        return super().get_all_masked_mean(array)

    def save_values(self, values, out_path):
        data = self.data.astype(np.float32)
        ll = self.labels.tolist()

        for k, v in values.items():
            data[data==k] = v
            ll.remove(k)
        for i in ll:
            data[data==i] = 0
        utils.gen_nii(data, self.nii, out_path)

class GiiMask(Mask):
    def __init__(self, filepath):
        self.filepath = filepath
        self.gii = nib.load(filepath)
        array = self.gii.darrays[0].data
        filename = os.path.split(filepath)[1]
        super().__init__(filename, array)

    def get_all_masked_volume(self, gii):
        array = gii.darrays[0].data
        array = np.nan_to_num(array)
        array = np.reshape(array, newshape=self.shape)
        return super().get_all_masked_volume(array)

    def get_all_masked_mean(self, gii):
        array = gii.darrays[0].data
        array = np.nan_to_num(array)
        array = np.reshape(array, newshape=self.shape)
        return super().get_all_masked_mean(array)
