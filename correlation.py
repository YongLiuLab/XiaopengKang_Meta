# %%
# Effect size with MMSE t-value
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import seaborn as sns

from numpy.polynomial.polynomial import polyfit
from scipy.stats import pearsonr

import datasets
import utils
from meta_analysis.main import csv_meta_analysis
import draw_results

plt.ioff()

class Result(object):
    def __init__(self, name, r, p):
        self.name = name
        self.r = r
        self.p = p
        
def pearson_r(name, values1_dict, values2_dict):
    v1 = []
    v2 = []
    for key, value1 in values1_dict.items():
        try:
            v2.append(values2_dict[key])
            v1.append(value1)
        except KeyError:
            pass
    r, p = pearsonr(v1, v2)
    return Result(name, r, p)

def cor_roi_confound(roi_models, confound_model, mask,
                     out_dir):
    confound_effect_sizes = confound_model.effect_sizes
    rs = {}
    ps = {}
    results = []
    for key, roi_model in roi_models.items():
        roi_effect_sizes = roi_model.effect_sizes
        r = pearsonr(confound_effect_sizes, roi_effect_sizes)[0]
        p = pearsonr(confound_effect_sizes, roi_effect_sizes)[1]
        key = int(key)
        rs[key] = r
        ps[key] = p
        results.append(Result(key, r, p))

    nii_array = mask.data.astype(np.float32)
    p_array = mask.data.astype(np.float32)
    ll = [i for i in range(1, 247)]
    for (k, r), (_, p) in zip(rs.items(), ps.items()):
        nii_array[nii_array==k] = r
        p_array[p_array==k] = p
        ll.remove(k)
    for i in ll:
        nii_array[nii_array==np.float32(i)] = 0

    path = os.path.join(out_dir, 'r.nii')
    p_path = os.path.join(out_dir, 'p.nii')
    utils.gen_nii(nii_array, mask.nii, path)
    utils.gen_nii(p_array, mask.nii, p_path)
    return results

# Correlation with PET
def cor_roi_pet(roi_models, pet_dir,
                fig_width=5, fig_height=5,
                out_dir=None, show=False, save=True,
                fontsize=18):
    files = os.listdir(pet_dir)
    roi_es_dict = {}
    for k, v in roi_models.items():
        roi_es_dict[k] = v.total_effect_size

    roi_df = pd.DataFrame.from_dict(roi_es_dict, orient='index', columns=['es'])
    roi_df.index.name = 'ID'
    roi_df.index = roi_df.index.map(int)

    results = []
    for f in files:
        path = os.path.join(pet_dir, f)
        df = pd.read_csv(path, index_col=0)
        nndf = pd.merge(roi_df, df, left_on='ID', right_on='ID')

        x = nndf['es'].to_list()
        y = nndf['Volume'].to_list()

        r = pearsonr(x, y)[0]
        p = pearsonr(x, y)[1]
        if 'SERT' in f:
            result = Result(f[:f.rfind('_')], r, p)
        else:
            result = Result(f[:f.find('_')], r, p)
        results.append(result)

        draw_results.plot_correlation_joint(x, y,
                'Effect sizes of ROI', f[:-4], fontsize=fontsize, show=show,
                save=save, out_path=os.path.join(out_dir, f[:-4]+'.png'))
    return results