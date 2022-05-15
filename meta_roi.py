#%%
def sort_models(models, orderby='es', descend=True):
    filenames = models.keys()
    models_list = models.values()
    if orderby == 'es':
        list1 = [model.total_effect_size for model in models_list]
    list1, models_list, filenames = (list(t) for t in zip(*sorted(zip(list1, models_list, filenames), reverse=descend)))
    return list1, dict(zip(filenames, models_list))

def bon_cor(models, thres=0.05):
    passed = {}
    not_passed = {}
    n = len(models)
    for name, model in models.items():
        if model.p * n <= thres:
            passed[name] = model
        else:
            not_passed[name] = model
    return passed, not_passed
#%%
# create csv for meta analysis
import datasets
import os
from meta_analysis import utils
import numpy as np
import csv

def create_csv_for_meta(centers, label_eg, label_cg,
                        csv_prefix, out_path='./data/meta_csv',
                        gender=None, ratio=1):
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    label_dir = os.path.join(out_path, '{}_{}'.format(label_eg, label_cg))
    if not os.path.isdir(label_dir):
        os.mkdir(label_dir)
    out_dir = os.path.join(label_dir, csv_prefix)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    mean_egs = []
    std_egs = []
    count_egs = []
    mean_cgs = []
    std_cgs = []
    count_cgs = []
    
    center_names = []
    for center in centers:
        if gender is None:
            persons_eg = center.get_by_label(label_eg)
            persons_cg = center.get_by_label(label_cg)
        else:
            persons_eg = center.get_by_gender_and_label(gender=gender,
                                                        label=label_eg)
            persons_cg = center.get_by_gender_and_label(gender=gender,
                                                        label=label_cg)
        features_eg, _, ids = center.get_csv_values(persons=persons_eg,
                                                    prefix=csv_prefix+'/{}.csv',
                                                    flatten=True)
        features_cg, *_ = center.get_csv_values(persons=persons_cg,
                                                prefix=csv_prefix+'/{}.csv',
                                                flatten=True)
        if features_eg is not None and features_cg is not None:
            # ratio controls how many subjects to include
            # For robust
            if ratio >1:
                ratio = 1
            np.random.shuffle(features_eg)
            np.random.shuffle(features_cg)
            features_eg = features_eg[:int(ratio*np.shape(features_eg)[0])]
            features_cg = features_cg[:int(ratio*np.shape(features_cg)[0])]

            mean_eg, std_eg, n_eg = utils.cal_mean_std_n(features_eg)
            mean_cg, std_cg, n_cg = utils.cal_mean_std_n(features_cg)
            mean_egs.append(mean_eg)
            std_egs.append(std_eg)
            count_egs.append(n_eg)
            mean_cgs.append(mean_cg)
            std_cgs.append(std_cg)
            count_cgs.append(n_cg)
            center_names.append(center.name)

    mean_egs = np.stack(mean_egs)
    std_egs = np.stack(std_egs)
    count_egs = np.stack(count_egs)
    mean_cgs = np.stack(mean_cgs)
    std_cgs = np.stack(std_cgs)
    count_cgs = np.stack(count_cgs)

    mean_egs_T = mean_egs.T
    std_egs_T = std_egs.T
    mean_cgs_T = mean_cgs.T
    std_cgs_T = std_cgs.T

    i = 0
    for ems, ess, cms, css,_id in zip(mean_egs_T, std_egs_T,
                                    mean_cgs_T, std_cgs_T,ids):
        i += 1
        csv_path = os.path.join(out_dir, '{}.csv'.format(_id))
        with open(csv_path, 'w', newline='') as file:
            fieldnames = ['center_name', 'm1','s1','n1', 'm2','s2','n2']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for em, es, en, cm, cs, cn, center_name in zip(
                ems, ess, count_egs, cms, css, count_cgs, center_names
            ):
                writer.writerow({'center_name': center_name,
                                'm1': em,
                                's1': es,
                                'n1': en,
                                'm2': cm,
                                's2': cs,
                                'n2': cn,})

# %%
import nibabel as nib
import pandas as pd
import numpy as np
import os
import utils
from nilearn import plotting
from meta_analysis.main import csv_meta_analysis
import correction
import pickle

pairs = [(2,0), (2,1), (1,0)]
nii_path  = './data/mask/rBN_Atlas_246_1mm.nii'
nii = nib.load(nii_path)
fea = 'roi_gmv_removed'

def meta_gmv(label_eg, label_cg, mask, save_nii=True,
             csv_prefix='roi_gmv_removed',
             csv_dir='./data/meta_csv',
             out_dir='./results/meta',
             count=246):
    models = {}
    prefix = '{}_{}/{}'.format(label_eg, label_cg, csv_prefix)
    csv_dir = os.path.join(csv_dir, prefix)
    out_dir = os.path.join(out_dir, prefix)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    csvs = os.listdir(csv_dir)
    
    for f in csvs:
        csv_path = os.path.join(csv_dir, f)
        model = csv_meta_analysis(csv_path, model_type='random')
        models[int(f[:-4])] = model

    if save_nii:
        nii_array = mask.data.astype(np.float32)
        p_array = mask.data.astype(np.float32)
        ll = mask.labels.tolist()

        for k, v in models.items():
            nii_array[nii_array==k] = v.total_effect_size
            p_array[p_array==k] = v.p
            ll.remove(k)
        for i in ll:
            nii_array[nii_array==i] = 0

        path = os.path.join(out_dir, 'es.nii')
        p_path = os.path.join(out_dir, 'p.nii')
        utils.gen_nii(nii_array, mask.nii, path)
        utils.gen_nii(p_array, mask.nii, p_path)
        correction.roi_correction(path, p_path, 246, out_dir)
    return models

# %%
import nilearn as nil
from nilearn import surface
from nilearn import plotting
import nibabel as nib
import pandas as pd
import os
from meta_analysis import utils
from nilearn import plotting
from meta_analysis.main import csv_meta_analysis
import numpy as np
from nibabel.gifti.gifti import GiftiDataArray,GiftiImage

pairs = [(2,0), (2,1),(1,0)]
annots = ['fsaverage.L.BN_Atlas.32k_fs_LR.label.gii', 'fsaverage.L.BN_Atlas.32k_fs_LR.label.gii']
surfs = ['lh.central.freesurfer.gii', 'rh.central.freesurfer.gii']
l_r = ['L', 'R']
csv_path = './data/mask/cortical_id.csv'
df = pd.read_csv(csv_path, index_col=0)
annot_dir = r'./data/mask/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/{}'
surf_dir = r'./data/mask/cat_surf_temp_fsavg32K/{}'

def meta_ct(label_eg, label_cg, p_thres=0.001,
            topn=0.3, save_gii=True,
            save_nii=False,
            mask=None,
            csv_prefix='roi_ct_removed',
            csv_dir_prefix='./data/meta_csv',
            out_dir_prefix='./results/meta', count=210):
    models = {}

    surfix = '{}_{}/{}'.format(label_eg, label_cg, csv_prefix)
    out_dir = os.path.join(out_dir_prefix, surfix)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    csv_dir = os.path.join(csv_dir_prefix, surfix)
    csvs = os.listdir(csv_dir)

    for f in csvs:
        csv_path = os.path.join(csv_dir, f)
        model = csv_meta_analysis(csv_path, model_type='random')
        models[int(f[:-4])] = model
    if save_gii:
        for annot, surf, lr in zip(annots, surfs, l_r):
            a = surface.load_surf_data(annot_dir.format(annot))
            a = a.astype(np.float32)
            b = surf_dir.format(surf)
            tmp_gii = nib.load(b)

            cor_model, _ = bon_cor(models, thres=p_thres)
            ll = np.unique(a).tolist()
            
            _, sorted_models = sort_models(cor_model, descend=False)
            top_es = sorted_models[int(len(sorted_models)*topn)].total_effect_size

            for k, v in cor_model.items():
                _id = np.float32(k)
                if v.total_effect_size <= top_es:
                    a[a==_id] = v.total_effect_size
                    if _id in ll:
                        ll.remove(_id)
            for i in ll:
                a[a==i] = 0
            
            gdarray = GiftiDataArray.from_array(a, intent=0)
            tmp_gii.remove_gifti_data_array_by_intent(0)
            tmp_gii.add_gifti_data_array(gdarray)
            path = os.path.join(out_dir, 'es_{}_bon{}_top{}.gii'.format(lr, str(p_thres)[2:], str(topn)[1:]))
            nib.save(tmp_gii, path)
    if save_nii:
        nii_array = mask.data.astype(np.float32)
        p_array = mask.data.astype(np.float32)

        ll = mask.labels.tolist()

        for k, v in models.items():
            nii_array[nii_array==k] = v.total_effect_size
            p_array[p_array==k] = v.p
            ll.remove(k)
        for i in ll:
            nii_array[nii_array==i] = 0

        path = os.path.join(out_dir, 'es.nii')
        p_path = os.path.join(out_dir, 'p.nii')
        utils.gen_nii(nii_array, mask.nii, path)
        utils.gen_nii(p_array, mask.nii, p_path)
        correction.roi_correction(path, p_path, 210, out_dir)
    return models

def create_csv_for_meta_asy(centers, label,
                        csv_prefix, out_path='./data/meta_csv/suppment/asy',
                        gender=None, template=None):
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    label_dir = os.path.join(out_path, '{}'.format(label))
    if not os.path.isdir(label_dir):
        os.mkdir(label_dir)
    out_dir = os.path.join(label_dir, csv_prefix)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    mean_egs = []
    std_egs = []
    count_egs = []
    mean_cgs = []
    std_cgs = []
    count_cgs = []
    
    center_names = []
    for center in centers:
        persons = center.get_by_label(label)

        features, _, ids = center.get_csv_values(persons=persons,
                                                 prefix=csv_prefix+'/{}.csv',
                                                 flatten=True)
        if features is not None:
             # GMV ROI size uneven
            if template:
                value = [np.sum(template.get_label_bool(label)) for label in template.labels]
                features = np.divide(features, value)
            features_l = features[:, ::2]
            features_r = features[:, 1::2]
            mean_eg, std_eg, n_eg = utils.cal_mean_std_n(features_l)
            mean_cg, std_cg, n_cg = utils.cal_mean_std_n(features_r)
            mean_egs.append(mean_eg)
            std_egs.append(std_eg)
            count_egs.append(n_eg)
            mean_cgs.append(mean_cg)
            std_cgs.append(std_cg)
            count_cgs.append(n_cg)
            center_names.append(center.name)

    mean_egs = np.stack(mean_egs)
    std_egs = np.stack(std_egs)
    count_egs = np.stack(count_egs)
    mean_cgs = np.stack(mean_cgs)
    std_cgs = np.stack(std_cgs)
    count_cgs = np.stack(count_cgs)

    mean_egs_T = mean_egs.T
    std_egs_T = std_egs.T
    mean_cgs_T = mean_cgs.T
    std_cgs_T = std_cgs.T

    i = 1
    for ems, ess, cms, css in zip(mean_egs_T, std_egs_T,
                                  mean_cgs_T, std_cgs_T):
        
        csv_path = os.path.join(out_dir, '{}.csv'.format(i))
        i += 2
        with open(csv_path, 'w', newline='') as file:
            fieldnames = ['center_name', 'm1','s1','n1', 'm2','s2','n2']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for em, es, en, cm, cs, cn, center_name in zip(
                ems, ess, count_egs, cms, css, count_cgs, center_names
            ):
                writer.writerow({'center_name': center_name,
                                'm1': em,
                                's1': es,
                                'n1': en,
                                'm2': cm,
                                's2': cs,
                                'n2': cn,})

def meta_gmv_asy(label, mask, save_nii=True,
                csv_prefix='roi_gmv_removed',
                csv_dir='./data/meta_csv/suppment/asy',
                out_dir='./results/supp/asym',
                count=123):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, str(label))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, csv_prefix)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    csv_dir = os.path.join(csv_dir, str(label), csv_prefix)
    csvs = os.listdir(csv_dir)

    models = {}
    for f in csvs:
        csv_path = os.path.join(csv_dir, f)
        model = csv_meta_analysis(csv_path, model_type='random')
        models[int(f[:-4])] = model

    if save_nii:
        nii_array = mask.data.astype(np.float32)
        p_array = mask.data.astype(np.float32)
        ll = mask.labels.tolist()

        for k, v in models.items():
            nii_array[nii_array==k] = v.total_effect_size
            p_array[p_array==k] = v.p
            ll.remove(k)
        for i in ll:
            nii_array[nii_array==i] = 0

        path = os.path.join(out_dir, 'es.nii')
        p_path = os.path.join(out_dir, 'p.nii')
        utils.gen_nii(nii_array, mask.nii, path)
        utils.gen_nii(p_array, mask.nii, p_path)
        correction.roi_correction(path, p_path, count, out_dir)
    return models