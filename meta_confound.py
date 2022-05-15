#%%
import datasets
import numpy as np
import scipy
import matplotlib.pyplot as plt
import meta_analysis
import os
import csv
from meta_analysis import utils
from meta_analysis.main import csv_meta_analysis

confounds = ['age', 'gmv', 'csf', 'wmv', 'tiv', 'MMSE', 'Resultion', 'Noise', 
            'Bias', 'IQR',]
# Create csv
def create_csv_for_meta(centers, label_eg, label_cg, confounds=confounds,
                        csv_dir='./data/meta_csv/{}_{}/confound/'):
    csv_dir = csv_dir.format(label_eg, label_cg)
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)
    for confound in confounds:
        csv_path = csv_dir + confound + '.csv'
        with open(csv_path, 'w', newline='') as file:
            fieldnames = ['center_name', 'm1','s1','n1', 'm2','s2','n2']
            #fieldnames = ['center_name', 'a','c', 'b', 'd']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for center in centers:
                if confound == 'age':
                    eg_features, *_ = center.get_ages(label_eg)
                    cg_features, *_ = center.get_ages(label_cg)
                    if eg_features is not None and cg_features is not None:
                        eg_features = eg_features * 100
                        cg_features = cg_features * 100
                elif confound == 'gmv':
                    eg_features, *_ = center.get_gmvs(label_eg)
                    cg_features, *_ = center.get_gmvs(label_cg)
                elif confound == 'csf':
                    eg_features, *_ = center.get_csfs(label_eg)
                    cg_features, *_ = center.get_csfs(label_cg)
                elif confound == 'wmv':
                    eg_features, *_ = center.get_wmvs(label_eg)
                    cg_features, *_ = center.get_wmvs(label_cg)
                elif confound == 'tiv':
                    eg_features, *_ = center.get_tivs(label_eg)
                    cg_features, *_ = center.get_tivs(label_cg)
                elif confound == 'MMSE':
                    eg_features, *_ = center.get_MMSEs(label_eg)
                    cg_features, *_ = center.get_MMSEs(label_cg)
                elif confound == 'Resolution':
                    eg_features, *_ = center.get_resolution(label_eg)
                    cg_features, *_ = center.get_resolution(label_cg)
                elif confound == 'Noise':
                    eg_features, *_ = center.get_noise(label_eg)
                    cg_features, *_ = center.get_noise(label_cg)
                elif confound == 'Bias':
                    eg_features, *_ = center.get_bias(label_eg)
                    cg_features, *_ = center.get_bias(label_cg)
                elif confound == 'IQR':
                    eg_features, *_ = center.get_iqr(label_eg)
                    cg_features, *_ = center.get_iqr(label_cg)
                if eg_features is not None and cg_features is not None:
                    mean_eg, std_eg, n_eg = utils.cal_mean_std_n(eg_features)
                    mean_cg, std_cg, n_cg = utils.cal_mean_std_n(cg_features)
                    writer.writerow({'center_name': center.name,
                                    'm1': mean_eg,
                                    's1': std_eg,
                                    'n1': n_eg,
                                    'm2': mean_cg,
                                    's2': std_cg,
                                    'n2': n_cg,})
# %%
def meta_confound(label_eg, label_cg,
                  csv_dir='./data/meta_csv/{}_{}/confound/',
                  output_dir='./results/meta/{}_{}/confound/'):
    models = {}
    csv_dir = csv_dir.format(label_eg, label_cg)
    output_dir = output_dir.format(label_eg, label_cg)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for confound in confounds:
        csv_path = csv_dir + confound + '.csv'
        output_path = output_dir + confound + '.png'
        model = csv_meta_analysis(csv_path, model_type='random')
        print(label_eg,label_cg,confound,model.p)
        model.plot_forest(title=confound, save_path=output_path, show=False)
        models[confound] = model
    return models