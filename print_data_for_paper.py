#%%
import datasets
import numpy as np

centers_adni = datasets.load_centers_adni()
centers_mcad = datasets.load_centers_mcad()
centers_edsd = datasets.load_centers_edsd()
centers_list = [centers_adni, centers_edsd, centers_mcad]
centers_name = ['ADNI', 'EDSD', 'MCAD']
labels = [0, 1, 2]

def get_all(centers, label):
    tmps = None
    for center in centers:
        # change func by demand
        
        tmp, _ = center.get_tivs_cgws(label)
        if tmp is not None:
            tmp = tmp[:,3]
        
        """
        tmp, _ = center.get_ages(label)
        if tmp is not None:
            tmp = tmp * 100
        """
        #tmp, _ = center.get_MMSEs(label)
        #tmp, _ = center.get_genders(label)
        if tmps is None:
            tmps = tmp
        else:
            if tmp is not None:
                tmps = np.concatenate([tmps, tmp])
    return tmps

for centers, name in zip(centers_list, centers_name):
    for label in labels:
        print(name, label)
        agess = get_all(centers, label)
        mean = np.mean(agess)
        std = np.std(agess)
        print('{:.2f}({:.2f})'.format(mean, std))
        #print('{}'.format(np.sum(1-agess)))
# %%
