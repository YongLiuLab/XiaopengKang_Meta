# %%
import os
import meta_analysis
import datasets
import nibabel as nib
from meta_analysis.mask import Mask
import numpy as np
from meta_analysis.main import voxelwise_meta_analysis
from meta_analysis import utils
from nibabel.gifti.gifti import GiftiDataArray,GiftiImage

temp_dir = r'./data/mask/cat_surf_temp_fsavg32K/{}'
surfs = ['lh.central.freesurfer.gii', 'rh.central.freesurfer.gii']
l_r = ['L', 'R']

def meta_gii(centers, label_eg, label_cg,
             out_dir='./results/meta/{}_{}'):
    out_dir = out_dir.format(label_eg, label_cg)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_dir = os.path.join(out_dir, 'surf')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    with open(os.path.join(out_dir, 'centers.txt'), "w") as text_file:
        center_mean_dict = {}
        center_std_dict = {}
        center_count_dict = {}
        for center in centers:
            n1 = len(center.get_by_label(label_eg))
            n2 = len(center.get_by_label(label_cg))
            if n1 == 0 or n2 == 0:
                continue
            print('{}: e:{}, c:{}'.format(center.name, n1, n2), file=text_file)
            group_mean_dict = {}
            group_std_dict = {}
            group_count_dict = {}
            for label in [label_eg, label_cg]:
                m, s, n = center.load_msn_array(label, _dir='surf')
                group_mean_dict[label] = m
                group_std_dict[label] = s
                group_count_dict[label] = n

            center_mean_dict[center.name] = group_mean_dict
            center_std_dict[center.name] = group_std_dict
            center_count_dict[center.name] = group_count_dict

    results = voxelwise_meta_analysis(label_eg, label_cg,
                                    center_mean_dict=center_mean_dict,
                                    center_std_dict=center_std_dict,
                                    center_count_dict=center_count_dict,
                                    dtype=np.float32)

    es = results[0]
    p = results[-1]

    es_l = es[:32492]
    es_r = es[32492:]
    p_l = p[:32492]
    p_r = p[32492:]

    es_l = es_l[p_l<0.001]
    es_r = es_r[p_r<0.001]

    path = os.path.join(out_dir, 'es_l_001.gii')
    gii_path = temp_dir.format('lh.central.freesurfer.gii')
    ct_gii = nib.load(gii_path)
    gdarray = GiftiDataArray.from_array(es_l, intent=0)
    ct_gii.remove_gifti_data_array_by_intent(0)
    ct_gii.add_gifti_data_array(gdarray)
    nib.save(ct_gii, path)
    
    path = os.path.join(out_dir, 'es_r_001.gii')
    gii_path = temp_dir.format('rh.central.freesurfer.gii')
    ct_gii = nib.load(gii_path)
    gdarray = GiftiDataArray.from_array(es_r, intent=0)
    ct_gii.remove_gifti_data_array_by_intent(0)
    ct_gii.add_gifti_data_array(gdarray)
    nib.save(ct_gii, path)

    
    result_names = ['es','var', 'se', 'll','ul','q','z','p']
    for result, name in zip(results, result_names):
        result_l = result[:32492]
        result_r = result[32492:]
        result_list = [result_l, result_r]

        for _result, surf, lr in zip(result_list, surfs, l_r):
            path = os.path.join(out_dir, '{}_{}.gii'.format(name, lr))
            gii_path = temp_dir.format(surf)
            ct_gii = nib.load(gii_path)
            gdarray = GiftiDataArray.from_array(_result, intent=0)
            ct_gii.remove_gifti_data_array_by_intent(0)
            ct_gii.add_gifti_data_array(gdarray)
            nib.save(ct_gii, path)
