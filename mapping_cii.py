import os
import time
import subprocess

import nibabel as nib
import numpy as np
import seaborn as sns
from matplotlib import colors, cm

def mapping(in_path,
            vloume_to_surface='{command_path} -volume-to-surface-mapping {in_path} {surf_path} {out_path} {mapping_method}',
            command_path='G:/workspace/HCP/workbench/bin_windows64/wb_command.exe',
            mapping_method='-enclosing',
            surf_path_L='G:/workspace/HCP/fsaverage/fsaverage_LR32k/fsaverage.L.inflated.32k_fs_LR.surf.gii',
            surf_path_R='G:/workspace/HCP/fsaverage/fsaverage_LR32k/fsaverage.R.inflated.32k_fs_LR.surf.gii'):
    """
    mapping_method: string, '-trilinear', '-enclosing', '-cubic'
    """
    _dir, fullname = os.path.split(in_path)
    filename, extension = os.path.splitext(fullname)
    out_path_L = os.path.join(_dir, filename+'.L.func.gii')
    out_path_R = os.path.join(_dir, filename+'.R.func.gii')

    cmd_L = vloume_to_surface.format(command_path=command_path, in_path=in_path,
                                   surf_path=surf_path_L, out_path=out_path_L,
                                   mapping_method=mapping_method)
    subp_L = subprocess.Popen(cmd_L)

    cmd_R = vloume_to_surface.format(command_path=command_path, in_path=in_path,
                                    surf_path=surf_path_R, out_path=out_path_R,
                                    mapping_method=mapping_method)
    subp_R = subprocess.Popen(cmd_R)
    return subp_L, subp_R, out_path_L, out_path_R

def readin_gii_data(path):
    return nib.load(path).darrays[0].data

def generate_mappable(cmap, vmin, vmax):
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    return mappable

def generate_cii(data, label_dict, brain_model_axis, out_path=None):
    label_axis = nib.cifti2.cifti2_axes.LabelAxis(['aaa'], label_dict)
    header = nib.cifti2.cifti2.Cifti2Header.from_axes((label_axis, brain_model_axis))
    image = nib.cifti2.cifti2.Cifti2Image(dataobj=data.reshape([1, -1]), header=header)
    if out_path:
        image.to_filename(out_path)
    return image

def get_max_min(data):
    if data.size > 0:
        vmin = np.min(data)
        vmax = np.max(data)
    else:
        vmin = 0
        vmax = 0
    return vmin, vmax

def get_min_max_all(in_pathes):
    v_min = 1000000
    v_max = -1000000
    for path in in_pathes:
        array = np.array(nib.load(path).dataobj)
        _min = np.min(array)
        _max = np.max(array)
        print(_min, _max)
        if _min < v_min:
            v_min = _min
        if _max > v_max:
            v_max = _max
        
    return v_min, v_max

def gmv_to_cii(in_path, in_path2=None,
                cmap_pos=None, vmin_pos=None, vmax_pos=None, mappable_pos=None,
                cmap_neg=None, vmin_neg=None, vmax_neg=None, mappable_neg=None,
                patience=10):
    # map volume nii to surface gii
    _dir, fullname = os.path.split(in_path)
    filename, extension = os.path.splitext(fullname)
    if extension == '.nii':
        sp1, sp2, out_path_L, out_path_R = mapping(in_path)
        
        complete = False
        for i in range(patience):
            if sp1.poll() == 0 and sp2.poll() == 0:
                complete = True
                break
            time.sleep(5)
            
        if not complete:
            raise ValueError('Mapping Error')
        # Readin surf data
        data_L = readin_gii_data(out_path_L)
        data_R = readin_gii_data(out_path_R)
    elif extension == '.gii' and in_path2 is not None:
        data_L = nib.load(in_path).darrays[2].data
        data_R = nib.load(in_path2).darrays[2].data

    data = np.concatenate([data_L, data_R])
    # scale to fit mappable
    data = np.nan_to_num(data)
    data = data * 100
    
    v_min, v_max = get_max_min(data[data<0])
    if vmin_neg is None:
        vmin_neg = v_min
    if vmax_neg is None:
        vmax_neg = v_max
    v_min, v_max = get_max_min(data[data>0])
    if vmin_pos is None:
        vmin_pos = v_min
    if vmax_pos is None:
        vmax_pos = v_max

    print('pos: {:.3f} {:.3f}'.format(vmin_pos, vmax_pos))
    print('neg: {:.3f} {:.3f}'.format(vmin_neg, vmax_neg))

    # generate color mapping
    if mappable_pos is None:
        if cmap_pos is None:
            raise ValueError('At least one of #cmap_pos and #mappable_pos should be specify')
        else:
            mappable_pos = generate_mappable(cmap_pos, vmin_pos, vmax_pos)

    if mappable_neg is None:
        if cmap_neg is None:
            raise ValueError('At least one of #cmap_neg and #mappable_neg should be specify')
        else:
            mappable_neg = generate_mappable(cmap_neg, vmin_neg, vmax_neg)

    # apply color mapping
    label_dict = {}
    for i, p in enumerate(data):
        if p > 0:
            color_value = mappable_pos.to_rgba(p)
        elif p < 0:
            color_value = mappable_neg.to_rgba(p)
        else:
            color_value = (1,1,1,1)
        label_dict[p] = (i, color_value)

    names = ['CIFTI_STRUCTURE_CORTEX_LEFT' for i in range(data_L.shape[0])]
    names.extend(['CIFTI_STRUCTURE_CORTEX_RIGHT' for i in range(data_L.shape[0])])
    verteces = [i for i in range(data_L.shape[0])]
    verteces.extend([i for i in range(data_L.shape[0])])
    verteces = np.asarray(verteces)
    brain_model_axis = nib.cifti2.cifti2_axes.BrainModelAxis(name=names, vertex=np.asarray(verteces),
                                                           nvertices={'CIFTI_STRUCTURE_CORTEX_LEFT': 32492,
                                                                      'CIFTI_STRUCTURE_CORTEX_RIGHT': 32492},)

    
    out_path = os.path.join(_dir, filename+'.dlabel.nii')

    cii = generate_cii(data, label_dict, brain_model_axis, out_path=out_path)
    return cii, mappable_pos, mappable_neg

def ct_to_cii(in_path,
              cmap_pos=None, vmin_pos=None, vmax_pos=None, mappable_pos=None,
              cmap_neg=None, vmin_neg=None, vmax_neg=None, mappable_neg=None,
              dlabel_path='G:/workspace/HCP/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii',
              temp_path = 'G:/workspace/HCP/rBN_Atlas_246_1mm.nii'):
    # Readin data 
    data = nib.load(in_path).get_fdata()
    print(in_path)
    v_min, v_max = get_max_min(data[data<0])
    if vmin_neg is None:
        vmin_neg = v_min
    if vmax_neg is None:
        vmax_neg = v_max
    v_min, v_max = get_max_min(data[data>0])
    if vmin_pos is None:
        vmin_pos = v_min
    if vmax_pos is None:
        vmax_pos = v_max

    print('pos: {:.3f} {:.3f}'.format(vmin_pos, vmax_pos))
    print('neg: {:.3f} {:.3f}'.format(vmin_neg, vmax_neg))

    # generate color mapping
    if mappable_pos is None:
        if cmap_pos is None:
            raise ValueError('At least one of #cmap_pos and #mappable_pos should be specify')
        else:
            mappable_pos = generate_mappable(cmap_pos, vmin_pos, vmax_pos)

    if mappable_neg is None:
        if cmap_neg is None:
            raise ValueError('At least one of #cmap_neg and #mappable_neg should be specify')
        else:
            mappable_neg = generate_mappable(cmap_neg, vmin_neg, vmax_neg)

    
    # Readin surf template
    dlable = nib.load(dlabel_path)
    dlabel_data = np.asarray(dlable.dataobj)
    label_axis = dlable.header.get_axis(0)
    label_dict = label_axis.get_element(0)[1]

    # init all roi:color pair
    for i in range(1,421):
        label_dict[i] = (label_dict[i][0], (1, 1, 1, 1))

    # Readin roi mask, perform extract color mapping for each roi
    temp_nii = nib.load(temp_path)
    temp = np.nan_to_num(temp_nii.get_fdata())

    for roi in range(1,211):
        fvalue = data[temp == roi][0]
        if fvalue > 0:
            color_value = mappable_pos.to_rgba(fvalue)
        elif fvalue < 0:
            color_value = mappable_neg.to_rgba(fvalue)
        else:
            color_value = (1,1,1,1)

        label_dict[roi] = (label_dict[roi][0], color_value)
        label_dict[roi+210] = (label_dict[roi+210][0], color_value)

    _dir, fullname = os.path.split(in_path)
    filename, extension = os.path.splitext(fullname)
    out_path = os.path.join(_dir, filename+'.dlabel.nii')

    cii = generate_cii(dlabel_data, label_dict, dlable.header.get_axis(1), out_path=out_path)
    return cii, mappable_pos, mappable_neg

