import datasets

import nibabel as nib
import os
import pickle
import pandas as pd
import seaborn as sns
import numpy as np

import draw_results
import gene_analysis
import removing_confound
import meta_roi
import meta_voxel
import meta_vertex
import meta_confound
import correlation
from mask import Mask, NiiMask
import mixed_lm

import datasets
import os
import mask
from multiprocessing import Process
import pickle

# Create ROI GMV file
def create_meta(centers, mask, iss):
    # Take 80% subjects for meta then performed PLSR
    label_eg = 2
    label_cg = 0
    for i in iss:
        print(i)
        meta_roi.create_csv_for_meta(centers, label_eg, label_cg,
                                    csv_prefix='roi_gmv_removed',
                                    out_path=os.path.join('./data/meta_csv/robust', str(i)),
                                    ratio=0.8)
        roi_gmv_models = meta_roi.meta_gmv(label_eg, label_cg, mask,
                                        csv_dir=f'./data/meta_csv/robust/{i}',
                                        out_dir='./results/robust',
                                        save_nii=False)
        with open(f'./results/robust/result_models_{i}.pkl', 'wb') as f:
            pickle.dump(roi_gmv_models, f)

if __name__ == '__main__':
    #Assume all data is organized
    #Load dataset
    centers = datasets.load_centers_all()
    #Define ROI Mask
    mask_path = './data/mask/rBN_Atlas_246_1mm.nii'
    mask = NiiMask(mask_path)

    iss = [i for i in range(5000)]
    chunks = [iss[x:x+500] for x in range(0, 5000, 500)]

    for j in range(10):
        p = Process(target=create_meta, args=(centers, mask, chunks[j],))
        p.start()
