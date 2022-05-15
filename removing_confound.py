#%%
import center
import datasets
import numpy as np
import mask
import nibabel as nib
from sklearn.preprocessing import OneHotEncoder

centers = datasets.load_centers_all()

def get_index(lst, item):
    return [i for i in range(len(lst)) if lst[i] > item]

def onehot(n_labels, labels):
    return np.eye(n_labels)[labels]

def minmax(x, axis=0):
    _min = np.min(x, axis=axis)
    _max = np.max(x, axis=axis)
    return (x - _min) / (_max - _min)

def z_norm(x, axis=0):
    _mean = np.mean(x, axis=axis)
    _std = np.std(x, axis=axis)
    return (x - _mean) / _std

def create_x(center, intercept=1, add_label=True, z_tran_TIV=True):
    # age, male, female
    amfms, labels = center.get_presonal_info_values()
    amfs = np.reshape(amfms[:,0:3],(-1,3))
    onehot_lables = onehot(3, labels)
    tcgws, *_ = center.get_tivs_cgws()
    tivs = np.reshape(tcgws[:,0],(-1,1))
    if z_tran_TIV:
        tivs = z_norm(tivs)
    if intercept:
        intercepts = np.reshape(np.repeat(intercept, len(labels)),(-1, 1))
        if add_label:
            x = np.hstack((amfs, tivs, intercepts, onehot_lables))
        else:
            x = np.hstack((amfs, tivs, intercepts))
    else:
        if add_label:
            x = np.hstack((amfs, tivs, onehot_lables))
        else:
            x = np.hstack((amfs, tivs))
    return x

def create_y_nii(center, mask, nii_prefix='mri_smoothed/{}.nii'):
    y = []
    d = mask.get_mask_data().flatten()
    index = get_index(d, 0)
    pathes, *_ = center.get_nii_pathes(nii_prefix=nii_prefix)
    for path in pathes:
        nii = nib.load(path)
        y.append(np.asarray(nii.dataobj, dtype=np.float16).flatten()[index])
    y = np.array(y)
    return y, index
#%%
#Remove Nii(persudo inverse)
"""
nii_prefix = 'mri_smoothed/{}.nii'
for center in centers:
    if len(center.persons) > 20:
        x = create_x(center, 1)
        y, index = create_y_nii(center, _mask, nii_prefix)
        x_inv = np.linalg.pinv(x)
        beta = np.dot(x_inv, y)
        beta_a = beta[:4]
        y_hats = np.dot(x[:,:4], beta_a)
        pathes, *_ = center.get_nii_pathes(nii_prefix=nii_prefix)
        for (person, path, yi, yi_hat) in zip(center.persons, pathes, y, y_hats):
            onii = nib.load(path)
            header = onii.header
            header.set_data_dtype(np.float32)

            image = np.zeros(shape=(181*217*181))
            for i in range(len(index)):
                image[index[i]] = yi[i] - yi_hat[i]
            image = np.reshape(image, (181, 217, 181))
            nii = nib.Nifti1Image(image, onii.affine, header)
            center.save_nii(person, nii)
#%%
"""
# Remove Nii using linear regression
from sklearn.linear_model import HuberRegressor
import time

def nii_regs(center, mask, nii_prefix='mri_smoothed/{}.nii',
             batch_reg=False,
             batch_size=500000):
    d = mask.data.flatten()
    flatten_indices = get_index(d, 0)
    total_indices = len(flatten_indices)
    current = 0
    end = 0

    st = time.time()

    x = None
    print(center.name)
    print('loading x')
    x = create_x(center)
    et = time.time()
    print('x time: {}'.format(et-st))
    st = et

    regs = []
    if batch_reg:
        while end < total_indices:
            end = current + batch_size
            if end > total_indices:
                end = total_indices
            batch_indices = flatten_indices[current:end]
            current = end

            ys = []
            print('loading y')
            pathes, *_ = center.get_nii_pathes(nii_prefix=nii_prefix)
            for path in pathes:
                nii = nib.load(path)
                ys.append(np.asarray(nii.dataobj).flatten()[batch_indices])
            et = time.time()
            print('y time: {}'.format(et-st))
            st = et
            ys = np.asarray(ys)
            print('Regression')
            for y in ys.T:
                reg = HuberRegressor().fit(x, y)
                regs.append(reg)
            et = time.time()
            print('Now:{}'.format(current))
            print('Regression time:{}'.format(et-st))
            st = et
    else:
        ys = []
        print('loading y')
        pathes, *_ = center.get_nii_pathes(nii_prefix=nii_prefix)
        for path in pathes:
            nii = nib.load(path)
            ys.append(np.asarray(nii.dataobj).flatten()[flatten_indices])
        et = time.time()
        print('y time: {}'.format(et-st))
        st = et
        ys = np.asarray(ys)
        print('Regression')
        i = 0
        for y in ys.T:
            reg = HuberRegressor().fit(x, y)
            regs.append(reg)
            if i % 500000 == 0:
                print('Now:{}'.format(i))
            i += 1
        et = time.time()
        print('Regression time:{}'.format(et-st))
        st = et
    print('Regs complete')
    return regs, flatten_indices

def save_removed_nii(center, regs, flatten_indices,
                     nii_prefix='mri_smoothed/{}.nii',
                     out_prefix='mri_smoothed_removed/{}.nii'):
    pathes, *_ = center.get_nii_pathes(nii_prefix=nii_prefix)
    x = create_x(center)
    for (path, xi, person) in zip(pathes, x, center.persons):
        onii = nib.load(path)
        header = onii.header
        header.set_data_dtype(np.float32)
        image = np.zeros(shape=(181*217*181))
        datas = np.asarray(onii.dataobj).flatten()[flatten_indices]
        for v, index, reg in zip(datas, flatten_indices, regs):
            image[index] = v - np.dot(xi[:4], reg.coef_[:4])
        image = np.reshape(image, (181, 217, 181))
        nii = nib.Nifti1Image(image, onii.affine, header)
        center.save_nii(person, nii, nii_prefix=out_prefix)

def remove_nii(centers, mask,
               nii_prefix='mri_smoothed/{}.nii',
               out_prefix='mri_smoothed_removed/{}.nii',
               batch_reg=False, batch_size=4000000):
    for center in centers:
        print(center.name)
        regs, flatten_indices = nii_regs(center, mask, nii_prefix,
                                         batch_reg=batch_reg, batch_size=batch_size)
        print('Saving data')
        save_removed_nii(center, regs, flatten_indices, nii_prefix, out_prefix)
        print('Saved')
#%%
# Remove Resampled gii using linear regression
from sklearn.linear_model import HuberRegressor
import time
from nibabel.gifti.gifti import GiftiDataArray,GiftiImage

def gii_regs(center):
    regs = []
    st = time.time()

    x = None
    print('loading x')
    x = create_x(center)
    et = time.time()
    print('x time: {}'.format(et-st))
    st = et

    ys = []
    print('loading y')
    pathes, *_ = center.get_cortical_thickness_pathes()
    for path in pathes:
        ct_gii = nib.load(path)
        ct_darray = ct_gii.get_arrays_from_intent(0)[0]
        ys.append(ct_darray.data.flatten())
    et = time.time()
    print('y time: {}'.format(et-st))
    st = et
    ys = np.asarray(ys)
    ys = np.nan_to_num(ys)
    print('Regression')
    for y in ys.T:
        reg = HuberRegressor().fit(x, y)
        regs.append(reg)
    et = time.time()
    print('Regression time:{}'.format(et-st))
    return regs

def remove_gii(centers):
    for center in centers:
        regs = gii_regs(center)

        pathes, *_ = center.get_cortical_thickness_pathes()
        x = create_x(center)
        for xi,path in zip(x, pathes):
            ct_gii = nib.load(path)
            newpath = path.replace('resampled_32k', 'resampled_32k.removed')
            ct_darray = ct_gii.get_arrays_from_intent(0)[0]
            data = ct_darray.data
            shape = data.shape
            data = np.nan_to_num(data)
            data = data.flatten()
            new_data = np.zeros_like(data)
            index = data!=np.nan
            for (reg,i) in zip(regs,index):
                new_data[i] = data[i] - np.dot(xi[:4], reg.coef_[:4])
            new_data = np.reshape(new_data, newshape=shape)
            gdarray = GiftiDataArray.from_array(new_data, intent=0)
            ct_gii.remove_gifti_data_array_by_intent(0)
            ct_gii.add_gifti_data_array(gdarray)
            nib.save(ct_gii, newpath)
"""
#%%
# Process and save removed Nii(Linear regression)


#%%
nii_prefix = 'mri_smoothed/{}.nii'
for center in centers:
    if len(center.persons) > 20:
        x = create_x(center, 1)
        y, index = create_y_nii(center, _mask, nii_prefix)
        x_inv = np.linalg.pinv(x)
        beta = np.dot(x_inv, y)
        beta_a = beta[:4]
        y_hat = np.dot(x[:,:4], beta_a)
        pathes, *_ = center.get_nii_pathes(nii_prefix=nii_prefix)
        for (person, path, yi, yi_hat) in zip(center.persons, pathes, y, y_hat):
            onii = nib.load(path)
            header = onii.header
            header.set_data_dtype(np.float32)

            image = np.zeros(shape=(181*217*181))
            for i in range(len(index)):
                image[index[i]] = yi[i] - yi_hat[i]
            image = np.reshape(image, (181, 217, 181))
            nii = nib.Nifti1Image(image, onii.affine, header)
            center.save_nii(person, nii)
"""
#%%
#--------------------------------------------------------------
# remove csv feature by center
from sklearn.linear_model import HuberRegressor
import csv
import os
def remove_roi(centers, csv_prefix='roi_gmv/{}.csv',
               out_prefix='roi_gmv_removed/{}.csv'):
    for center in centers:
        print(center.name)
        print('loading x')
        x = create_x(center)
        print('loading y')
        ys, _, ids = center.get_csv_values(prefix=csv_prefix, flatten=True)
        yst = ys.T
        regs = []
        print('Regression')
        for y in yst:
            reg = HuberRegressor().fit(x, y)
            regs.append(reg)
        print('Write New Data')
        for (xi, person, yi) in zip(x, center.persons, ys):
            xi = xi[:4]
            path = os.path.join(center.file_dir, out_prefix.format(person.filename))
            with open(path, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['ID', 'GMV'])
                writer.writeheader()
                for (yii, reg, _id) in zip(yi, regs, ids):
                    yii_hat = yii - np.dot(xi, reg.coef_[:4])
                    writer.writerow({'ID': _id,
                                    'GMV': yii_hat})

"""
# %%
from sklearn.linear_model import HuberRegressor
import csv
import os
csv_prefix = 'roi_ct/{}.csv'
out_prefix = 'roi_ct_removed/{}.csv'

x = None
print('loading x')
for center in centers:
    if x is None:
        x = create_x(center)
    else:
        x = np.concatenate((x, create_x(center))) 
yss = None
for center in centers:
    ys, _, ids = center.get_csv_values(prefix=csv_prefix, flatten=True)
    if yss is None:
        yss = ys
    else:
        yss = np.concatenate((yss, ys)) 
yst = yss.T
regs = []
for y in yst:
    reg = HuberRegressor().fit(x, y)
    regs.append(reg)

for center in centers:
    if len(center.persons) > 20:
        x = create_x(center)
        ys, _, ids = center.get_csv_values(prefix=csv_prefix, flatten=True)
        for (xi, person, yi) in zip(x, center.persons, ys):
            xi = xi[:4]
            path = os.path.join(center.file_dir, out_prefix.format(person.filename))
            with open(path, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['ID', 'CT'])
                writer.writeheader()
                for (yii, reg, _id) in zip(yi, regs, ids):
                    yii_hat = yii - np.dot(xi, reg.coef_[:4])
                    writer.writerow({'ID': _id,
                                     'CT': yii_hat})

# %%
"""