#%%
import datasets
import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd

def write_center_count(csv_path='./data/center_info/count.csv'):
    centers = datasets.load_centers_all()

    with open(csv_path, 'w', newline='') as file:
        fieldnames = ['center', 'NC', 'MC', 'AD']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for center in centers:
            n = len(center.get_by_label(0))
            m = len(center.get_by_label(1))
            a = len(center.get_by_label(2))
            writer.writerow({'center': center.name,
                             'NC': n,'MC':m,'AD':a})
            
def get_count(series, study):
    tmp = 0
    for index, value in series.iteritems():
        if study in index:
            tmp += value
    return tmp
            
def pie_plot(csv_path='./data/center_info/count.csv',
         radius = 1.5, size = 0.55,
         text_weight='bold', text_size=12,
         inside_labeldistance=0.53,
         outside_pctdistance=0.82):
    df = pd.read_csv(csv_path, index_col=0)

    nc = df['NC']
    mci = df['MC']
    ad = df['AD']

    nc_adni = get_count(nc, 'ADNI')
    nc_mcad = get_count(nc, 'MCAD')
    nc_edsd = get_count(nc, 'EDSD')
    mc_adni = get_count(mci, 'ADNI')
    mc_mcad = get_count(mci, 'MCAD')
    mc_edsd = get_count(mci, 'EDSD')
    ad_adni = get_count(ad, 'ADNI')
    ad_mcad = get_count(ad, 'MCAD')
    ad_edsd = get_count(ad, 'EDSD')

    adni = [nc_adni, mc_adni, ad_adni]
    edsd = [nc_edsd, mc_edsd, ad_edsd]
    mcad = [nc_mcad, mc_mcad, ad_mcad]

    nc = [nc_adni, nc_edsd, nc_mcad]
    mc = [mc_adni, mc_edsd, mc_mcad]
    ad = [ad_adni, ad_edsd, ad_mcad]
    fig, ax = plt.subplots()

    vals = np.array([adni, edsd, mcad])
    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.arange(3)*4)
    inner_colors = cmap(np.array([1,2,3,5,6,7,9,10,11]))

    textprops = {'family' : 'normal',
                    'weight' : text_weight,
                    'size'   : text_size}

    ax.pie(vals.flatten(), 
           labels=['NC', 'MCI', 'AD']*3,
           autopct='%.2f%%',
           pctdistance=outside_pctdistance,
           textprops=textprops,
           radius=radius, colors=inner_colors,
           wedgeprops=dict(width=size, edgecolor='w'))
    ax.pie(vals.sum(axis=1),
           labels=['ADNI', 'EDSD', 'MCAD'],
           labeldistance=inside_labeldistance,
           textprops=textprops,
           radius=radius-size, colors=outer_colors,
           wedgeprops=dict(width=size, edgecolor='w'))
    plt.show()

#%%
def get_all(centers, label):
    males = 0
    females = 0
    ages = None
    mmses = None
    csfs = None
    gmvs = None
    wmvs = None
    tivs = None

    for center in centers:
        tcgw, _ = center.get_tivs_cgws(label)
        if tcgw is not None and tcgw.size != 0:
            tiv = tcgw[:, 0]
            csf = tcgw[:, 1]
            gmv = tcgw[:, 2]
            wmv = tcgw[:, 3]
        else:
            tiv = None
            csf = None
            gmv = None
            wmv = None
        age, _ = center.get_ages(label)
        mmse, _ = center.get_MMSEs(label)
        male, _ = center.get_males(label)
        female, _ = center.get_females(label)

        if ages is None and age is not None:
            ages = age * 100
        else:
            if age is not None:
                ages = np.concatenate([ages, age * 100])
        if mmses is None:
            mmses = mmse
        else:
            if mmse is not None:
                mmses = np.concatenate([mmses, mmse])
        if csfs is None:
            csfs = csf
        else:
            if csf is not None:
                csfs = np.concatenate([csfs, csf])
        if gmvs is None:
            gmvs = gmv
        else:
            if gmv is not None:
                gmvs = np.concatenate([gmvs, gmv])
        if wmvs is None:
            wmvs = wmv
        else:
            if wmv is not None:
                wmvs = np.concatenate([wmvs, wmv])
        if tivs is None:
            tivs = tiv
        else:
            if tiv is not None:
                tivs = np.concatenate([tivs, tiv])
        
        if male is not None:
            males += np.sum(male)
            females += np.sum(female)
    return males, females, ages, mmses, csfs, gmvs, wmvs, tivs

def get_center_stats(csv_path='./results/s.csv'):
    centers_mcad = datasets.load_centers_mcad()
    centers_adni = datasets.load_centers_adni()
    centers_edsd = datasets.load_centers_edsd()
    centers_list = [centers_mcad, centers_adni, centers_edsd]
    centers_name = ['MCAD', 'ADNI', 'EDSD']
    labels = [0, 1, 2]

    maless = []
    femaless = []
    ages_mean = []
    ages_std = []
    mmses_mean = []
    mmses_std = []
    csfs_mean = []
    csfs_std = []
    gmvs_mean = []
    gmvs_std = []
    wmvs_mean = []
    wmvs_std = []
    tivs_mean = []
    tivs_std = []

    for centers, name in zip(centers_list, centers_name):
        for label in labels:
            print(name, label)
            males, females, ages, mmses, csfs, gmvs, wmvs, tivs = get_all(centers, label)
            maless.append(males)
            femaless.append(females)
            ages_mean.append(np.mean(ages))
            ages_std.append(np.std(ages))
            mmses_mean.append(np.mean(mmses))
            mmses_std.append(np.std(mmses))
            csfs_mean.append(np.mean(csfs))
            csfs_std.append(np.std(csfs))
            gmvs_mean.append(np.mean(gmvs))
            gmvs_std.append(np.std(gmvs))
            wmvs_mean.append(np.mean(wmvs))
            wmvs_std.append(np.std(wmvs))
            tivs_mean.append(np.mean(tivs))
            tivs_std.append(np.std(tivs))

    df = pd.DataFrame({'Male':maless, 'Female':femaless,
                       'ages_mean':ages_mean, 'ages_std':ages_std,
                       'mmses_mean':mmses_mean, 'mmses_std':mmses_std,
                       'csfs_mean':csfs_mean, 'csfs_std':csfs_std,
                       'gmvs_mean':gmvs_mean, 'gmvs_std':gmvs_std,
                       'wmvs_mean':wmvs_mean, 'wmvs_std':wmvs_std,
                       'tivs_mean':tivs_mean, 'tivs_std':tivs_std})

    df.to_csv(csv_path)
    return df

def print_gender_count():
    labels = [0,1,2]
    centers_mcad = datasets.load_centers_mcad()
    centers_adni = datasets.load_centers_adni()
    centers_edsd = datasets.load_centers_edsd()
    centers_list = [centers_mcad, centers_adni, centers_edsd]
    for centers in centers_list:
        for label in labels:
            males = get_all(centers, label)
            print(label)
            print(len(males[males==1]))
            print(len(males[males==0]))