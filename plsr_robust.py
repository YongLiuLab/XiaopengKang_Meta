import datasets
import os
import mask
from multiprocessing import Process
import pickle

import gene_analysis
# Create ROI GMV file
def create_roi_volume_csv(iss):
    n_perm_boot = 1
    n_components = 5
    out_dir ='./results/robust/plsr'
    for i in iss:
        print(i)
        with open(f'./results/robust/result_models/result_models_{i}.pkl', 'rb') as f:
            roi_gmv_models = pickle.load(f)

            gmv_es_dict = {}
            for k,v in roi_gmv_models.items():
                gmv_es_dict[int(k)] = v.total_effect_size

            gmv_plsr = gene_analysis.plsr(gmv_es_dict, n_components=n_components,
                                        n_perm=n_perm_boot, n_boot=n_perm_boot,
                                        out_path=os.path.join(out_dir, f'plsr_gmv_{i}.csv'))

if __name__ == '__main__':
    subjects = [i for i in range(5000)]
    chunks = [subjects[x:x+500] for x in range(0, 5000, 500)]

    for i in range(10):
        p = Process(target=create_roi_volume_csv, args=(chunks[i],))
        p.start()
