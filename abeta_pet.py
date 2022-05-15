#%%
# process abeta data
from scipy.io import loadmat
import numpy as np
import pandas as pd
from meta_analysis import model, data, utils
from scipy.stats import ttest_ind

def load_data():
    data_mat = loadmat('./data/PET/abeta/pet_value.mat')
    datas = data_mat['pet_value'].T
    return datas

def create_subject_df():
    name_mat = loadmat('./data/PET/abeta/subname.mat')
    info_mat = loadmat('./data/PET/abeta/info_corr.mat')
    olabels = info_mat['info'][0][0][-2].flatten()
    
    labels = []
    for l in olabels:
        if l:
            if l[0] == 'CN':
                labels.append(0)
            elif l[0] == 'MCI':
                labels.append(1)
            elif l[0] == 'Dementia':
                labels.append(2)
        else:
            labels.append(1)

    ages = info_mat['info'][0][0][0].flatten()
    ages = ages / 100
    mmse = info_mat['info'][0][0][1].flatten()
    ogender = info_mat['info'][0][0][4].flatten()
    male = np.array([1 if i == 1 else 0 for i in ogender])

    names = name_mat['subname'].flatten()
    names = [name[0] for name in names]
    centers = get_phases()

    subject_df = pd.DataFrame()
    subject_df['Name'] = names
    subject_df['center'] = centers
    subject_df['MMSE'] = mmse
    subject_df['Label'] = labels
    subject_df['male'] = male
    
    datas = load_data()
    for i, roi in zip(range(247), datas):
        subject_df[str(i+1)] = roi
    subject_df = subject_df[subject_df.center!='ADNIGO']
    return subject_df

def get_phases():
    name_mat = loadmat('./data/PET/abeta/subname.mat')
    names = name_mat['subname'].flatten()
    center_info = pd.read_csv('./data/center_info/ADNI/final_screening.csv', index_col=2)
    center_labels = []
    for name in names:
        phase = center_info.loc[name[0]]['COLPROT']
        center_labels.append(phase)
    return center_labels

def get_sites():
    name_mat = loadmat('./data/PET/abeta/subname.mat')
    names = name_mat['subname'].flatten()
    sites = []
    for name in names:
        sites.append(name[0][:3])
    return sites

def load_values(subject_df, label):
    sub_df = subject_df[subject_df.Label==label]
    sub_df = sub_df.iloc[:,5:]
    ids = list(sub_df.columns)
    ids = [int(_id) for _id in ids]
    return sub_df.values.T, ids

def ttest(a, b, ids, axis=0):
    ts, ps = ttest_ind(a, b, axis)
    t_dict = dict(zip(ids, ts))
    p_dict = dict(zip(ids, ps))
    return t_dict, p_dict

subject_df = create_subject_df()
def ttest_by_label(label1, label2, subject_df=subject_df):
    a, ids = load_values(subject_df, label1)
    b, _ = load_values(subject_df, label2)
    t, p = ttest(a.T, b.T, ids)
    return t, p

