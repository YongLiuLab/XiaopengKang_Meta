#%%
import os
import meta_analysis
import datasets
import nibabel as nib
import numpy as np
from meta_analysis.main import voxelwise_meta_analysis
from meta_analysis.mask import Mask
import utils
#%%
# load nii of mean, std, preform voxelwise_meta_analysis
def meta_nii(centers, label_eg, label_cg,
               mask_path='./data/mask/rBN_Atlas_246_1mm.nii',
               mri_dir='mri_smoothed_removed',
               out_dir='./results/meta/{}_{}'):
    # Generate mask instance
    mask_nii = nib.load(mask_path)
    mask = Mask(np.asarray(mask_nii.dataobj))
    
    # set out dir
    out_dir = out_dir.format(label_eg, label_cg)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_dir = os.path.join(out_dir, 'voxel')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    # save center info in centers.txt
    with open(os.path.join(out_dir, 'centers.txt'), "w") as text_file:
        # init empty center dict for mean, std, count 
        center_mean_dict = {}
        center_std_dict = {}
        center_count_dict = {}
        # for each center, create group dict for each label
        for center in centers:
            n1 = len(center.get_by_label(label_eg))
            n2 = len(center.get_by_label(label_cg))
            # check both label are not 0
            if n1 == 0 or n2 == 0:
                continue
            # print N to file
            print('{}: e:{}, c:{}'.format(center.name, n1, n2), file=text_file)
            # init empty group dict for mean, std, count
            group_mean_dict = {}
            group_std_dict = {}
            group_count_dict = {}
            for label in [label_eg, label_cg]:
                # get mean, std, count for certain label
                m, s, n = center.load_msn_nii_array(label, _dir=mri_dir)
                group_mean_dict[label] = m
                group_std_dict[label] = s
                group_count_dict[label] = n
            # save in center dict
            center_mean_dict[center.name] = group_mean_dict
            center_std_dict[center.name] = group_std_dict
            center_count_dict[center.name] = group_count_dict
    # perform voxelwise meta analysis
    results = voxelwise_meta_analysis(label_eg, label_cg,
                                    center_mean_dict=center_mean_dict,
                                    center_std_dict=center_std_dict,
                                    center_count_dict=center_count_dict,
                                    _mask=mask, dtype=np.float32)
    # save all results as nii
    result_names = ['es','var', 'se', 'll','ul','q','z','p']
    for result, name in zip(results, result_names):
        path = os.path.join(out_dir, '{}.nii'.format(name))
        utils.gen_nii(result, mask_nii, path)