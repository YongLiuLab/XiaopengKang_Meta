import math
from multiprocessing import Process
import os
from matplotlib import markers
from sklearn.metrics import zero_one_loss

from sklearn.model_selection import KFold
import datasets
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel, ConstantKernel
from gpr_kernel import ARD
import matplotlib.pyplot as plt
import pickle
import threading

# Construct Normative model (GPR) using NC data
# x: age, gender, tiv/act | y: ROI_i
# Loading NC 

def prepare_data(centers, prefix, target_label):
    center_names = []
    persons = []
    MMSEs = []
    ages = []
    genders = []
    tivs = []
    acts = []
    all_features = None
    for center in centers:
        persons_ = center.get_by_label(target_label)
        if persons_:
            center_names += [center.name for person in persons_]
            persons += [person.filename for person in persons_]
            MMSEs += center.get_MMSEs(target_label)[0].tolist()
            ages += center.get_ages(target_label)[0].tolist()
            genders += center.get_males(target_label)[0].tolist()
            tivs += center.get_tivs(target_label)[0].tolist()
            acts += center.get_average_thickness(target_label)[0].tolist()

            features, *_ = center.get_csv_values(persons=persons_, prefix=prefix, flatten=True)
            if all_features is None:
                all_features = features
            else:
                all_features = np.vstack((all_features, features))
    return center_names, persons, MMSEs, ages, genders, tivs, acts, all_features

def fit_gpr(rois, ages, genders, all_features, MMSEs, tivs, acts, save_dir):
    n_restarts_optimizer = 50
    for roi in rois:
        print(f'ROI {roi} started')
        x = []
        y = []
        for age, gender, rois, MMSE, act, tiv in zip(ages, genders, all_features, MMSEs, acts, tivs):
            x.append([age, gender])
            y.append([rois[roi]])
        x = np.array(x)
        y = np.array(y)

        #kernel =  DotProduct()+\
        kernel =  1 * RBF(length_scale=[1, 1], length_scale_bounds=(1e-10,1e10)) +\
                WhiteKernel(noise_level_bounds=(1e-10,1e10))

        gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
        gaussian_process.fit(x, y)
        with open(os.path.join(save_dir, f'model/gpr_{roi}.pkl'), 'wb') as f:
            pickle.dump(gaussian_process, f)

        r_square = gaussian_process.score(x, y)

        # plot Age
        arr1inds = x[:, 0].argsort()
        x_sorted = x[arr1inds[::-1]]
        mean_prediction, std_prediction = gaussian_process.predict(x_sorted, return_std=True)

        plt.scatter(x[:, 0], y, label=r"True", c='#2ca02c', s=2)
        plt.scatter(x_sorted[:, 0], mean_prediction, label="Mean prediction")
        plt.fill_between(
            x_sorted[:, 0].ravel(),
            mean_prediction.ravel() - 1.96 * std_prediction,
            mean_prediction.ravel() + 1.96 * std_prediction,
            alpha=0.5,
            label=r"95% confidence interval",
        )
        plt.xlabel("$Age$")
        plt.ylabel("$CT(Combat)$")
        plt.title(f'r2={r_square}')
        plt.savefig(os.path.join(save_dir, f'fig/train_{roi}_age.png'))
        plt.close()
        """
        # plot TIV
        arr1inds = x[:, 2].argsort()
        x_sorted = x[arr1inds[::-1]]
        mean_prediction, std_prediction = gaussian_process.predict(x_sorted, return_std=True)

        plt.scatter(x[:, 2], y, label=r"True", c='#2ca02c', s=2)
        plt.scatter(x_sorted[:, 2], mean_prediction, label="Mean prediction")
        plt.fill_between(
            x_sorted[:, 2].ravel(),
            mean_prediction.ravel() - 1.96 * std_prediction,
            mean_prediction.ravel() + 1.96 * std_prediction,
            alpha=0.5,
            label=r"95% confidence interval",
        )
        plt.xlabel("$TIV$")
        plt.ylabel("$CT(Combat)$")
        plt.title(f'r2={r_square}')
        plt.savefig(os.path.join(save_dir, f'fig/train_{roi}_tiv.png'))
        plt.close()
        """
        print(f'ROI {roi} ended')

if __name__ == '__main__':
    centers = datasets.load_centers_all()

    train_label = 0
    prefix = 'neurocombat_ct2/{}.csv'
    center_names, persons, MMSEs, ages, genders, tivs, acts, all_features = prepare_data(centers, prefix, train_label)

    save_dir = './results_0401/normative_model3/combat_ct/age_gender'

    #rois = np.arange(210)
    
    rois = [16,17,18,19,20,21,22,23,24,25,26,43,44,45,46,47,48,49,50,51,
            52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,
            72,73,74,75,76,77,78,79,80,96,97,98,99,100,101,102,103,104,
            105,106,107,124,125,126,127,128,129,130,131,132,133,134,150,
            151,152,153,154,155,156,157,158,159,160,161,178,179,180,181,
            182,183,184,185,186,187,188,192,193,194,195,196,197,198,199,
            200,201,202,203,204,205,206,207,208,209]
    
    process = 6
    n = len(rois)
    print(n)
    n_each = math.ceil(n/process)

    chunks = [rois[x:x+n_each] for x in range(0, len(rois), n_each)]
    for i in range(process):
        p = Process(target=fit_gpr, args=(chunks[i], ages, genders, all_features, MMSEs, tivs, acts, save_dir))
        p.start()