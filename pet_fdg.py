#%%
import pandas as pd
import shutil
import os
import csv
import nibabel as nib
import numpy as np
from scipy.stats import ttest_ind

def create_subject_df(path='./data/PET/fdg/nii', csv_path='./data/PET/fdg/all_info_fixed_age_flitered.csv'):
    files = os.listdir(path)
    df = pd.read_csv(csv_path, index_col='PTID')
    subject_df = pd.DataFrame(columns = ['Name', 'center', 'Label', 'MMSE', 'male', 'nii_path', 'sum_path'])
    for f in files:
        try:
            subject_name = f[:-4]
            row = df.loc[subject_name]
            age = row['TrueAge']
            site = row['COLPROT']
            label = row['Label']
            MMSE = row['MMSE']
            male = row['male']
            subject_df = subject_df.append({'Name' : subject_name, 'center' : site, 'Label' : label,
                                           'MMSE': MMSE, 'male': male, 'nii_path': os.path.join(path, f),
                                            'sum_path': os.path.join(path, 'sum', subject_name+'.csv')}, 
                                    ignore_index = True)
        except KeyError:
            pass
    return subject_df

def create_sum(subject_df, mask):
    for index, row in subject_df.iterrows():
        nii_path = row['nii_path']
        csv_path = row['sum_path']
        if not os.path.exists(csv_path):
            nii = nib.load(nii_path)
            volumes = mask.get_all_masked_volume(nii)

            with open(csv_path, 'w', newline='') as file:
                fieldnames = ['ID', 'GMV']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for k, v in volumes.items():
                    writer.writerow({'ID': k, 'GMV': v})

def load_values(subject_df, label):
    features = []
    for index, row in subject_df.iterrows():
        csv_path = row['sum_path']
        if os.path.exists(csv_path):
            if row['Label'] == label:
                df = pd.read_csv(csv_path, index_col=0)
                features.append(df.to_numpy().flatten())
                ids = df.index.tolist()
    features = np.stack(features)
    return features, ids

def ttest(a, b, ids, axis=0):
    ts, ps = ttest_ind(a, b, axis)
    t_dict = dict(zip(ids, ts))
    p_dict = dict(zip(ids, ps))
    return t_dict, p_dict

subject_df = create_subject_df()
def ttest_by_label(label1, label2, subject_df=subject_df):
    a, ids = load_values(subject_df, label1)
    b, _ = load_values(subject_df, label2)
    t, p = ttest(a, b, ids)
    return t, p