#%%
import os
import numpy as np
import nibabel as nib
from nibabel import nifti1
from scipy.stats import norm
import numpy as np
from meta_analysis import utils

def bonferroni_correction(array, p_array, count, p=0.05):
    thresed_p = p_array < p / count
    return np.multiply(array, thresed_p)

def load_nii_array(filepath):
    nii =  nib.load(filepath)
    return np.asarray(nii.dataobj), nii

def roi_correction(value_path, p_path, count, out_path, p=0.001, top=1):
    v_array, v_nii = load_nii_array(value_path)
    p_array, _ = load_nii_array(p_path)

    corrected_array = bonferroni_correction(v_array, p_array, count, p=p)
    unique = np.unique(corrected_array)
    sorted_unique = np.sort(unique)
    n = int(top*len(sorted_unique))
    thres = sorted_unique[n-1]
    corrected_array[corrected_array > thres] = 0
    cor_path = os.path.join(out_path, 'es_bon{}_top{}.nii'.format(str(p)[2:],
                                                                  int(top*100)))
    utils.gen_nii(corrected_array, v_nii, cor_path)
