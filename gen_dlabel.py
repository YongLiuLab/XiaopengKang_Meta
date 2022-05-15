#%%
import numpy as np
import mapping_cii
import seaborn as sns
from matplotlib import cm, colors
import matplotlib.pyplot as plt

colors1 = cm.gnuplot2(np.linspace(0, 1, 128))[50:]
cmap_neg = colors.LinearSegmentedColormap.from_list('colormap_neg', colors1)

colors2 = np.vstack((cm.BuGn(np.linspace(0, 1, 32)[8:])))
cmap_pos = colors.LinearSegmentedColormap.from_list('colormap_pos', colors2)

# combine them and build a new colormap
cs = np.vstack((colors1, colors2))
cmap = colors.LinearSegmentedColormap.from_list('my_colormap', cs)
# 绘制colormap
gradient = np.linspace(0, 1, 128)
gradient = np.vstack((gradient, gradient))
plt.imshow(gradient, aspect='auto', cmap=cmap)
#%%
# GMV
main_nii_path = r'H:\workspace\AD_meta\results\meta\2_0\roi_gmv_removed\es_bon001_top100.nii'
sub_nii_pathes = [r'H:\workspace\AD_meta\results\meta\2_1\roi_gmv_removed\es_bon001_top100.nii',
                   r'H:\workspace\AD_meta\results\meta\1_0\roi_gmv_removed\es_bon001_top100.nii']

_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path,
                                 vmax_neg=0, cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
#%%
# CT
main_nii_path = r'H:\workspace\AD_meta\results\meta\2_0\roi_ct_removed\es_bon001_top100.nii'
sub_nii_pathes = [r'H:\workspace\AD_meta\results\meta\2_1\roi_ct_removed\es_bon001_top100.nii',
                   r'H:\workspace\AD_meta\results\meta\1_0\roi_ct_removed\es_bon001_top100.nii']

_, mappable_pos, mappable_neg = mapping_cii.ct_to_cii(main_nii_path, cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.ct_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
# %%
# 绘制JuSpace示意图

juspace_path = r'H:\workspace\AD_meta\data\PET\5HT1a_WAY_HC36.nii'
cmap_neg = cm.get_cmap('gnuplot2_r')
mapping_cii.gmv_to_cii(juspace_path,
                        vmax_pos=0, vmax_neg=0,
                        vmin_pos=100, 
                        cmap_pos=cmap_neg, cmap_neg=cmap_neg,
                        patience=20)
# %%
# 绘制FDG示意图
fdg_path = r'H:\workspace\AD_meta\data\PET\fdg\nii\002_S_0295.nii'
cmap_neg = cm.get_cmap('gnuplot2')
mapping_cii.gmv_to_cii(fdg_path, vmax_neg=0, vmin_pos=0, cmap_pos=cmap_neg, cmap_neg=cmap_neg)
# %%
# 绘制MMSE t-map
gmv_path = r'E:\workspace\AD_meta\results\mixedLM\gmv_t.nii'
cmap_neg = cm.get_cmap('gnuplot2_r')
mapping_cii.gmv_to_cii(gmv_path, vmax_neg=0, vmin_pos=0, cmap_pos=cmap_neg, cmap_neg=cmap_neg)

ct_path = r'E:\workspace\AD_meta\results\mixedLM\ct_t.nii'
cmap_neg = cm.get_cmap('gnuplot2_r')
mapping_cii.ct_to_cii(ct_path, vmax_neg=0, vmin_pos=0, cmap_pos=cmap_neg, cmap_neg=cmap_neg)
# %%
# 绘制Voxel-wise
main_nii_path = r'H:\workspace\AD_meta\results\meta\2_0\voxel\es.nii'
sub_nii_pathes = [r'H:\workspace\AD_meta\results\meta\2_1\voxel\es.nii',
                   r'H:\workspace\AD_meta\results\meta\1_0\voxel\es.nii']

_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path,
                                            vmax_neg=0, vmin_pos=0,
                                            cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
# %%
# 绘制Vertex-wise
main_nii_path = [r'H:\workspace\AD_meta\results\meta\2_0\surf\es_L.gii', 
                r'H:\workspace\AD_meta\results\meta\2_0\surf\es_R.gii']
sub_nii_pathes = [[r'H:\workspace\AD_meta\results\meta\2_1\surf\es_L.gii', 
                    r'H:\workspace\AD_meta\results\meta\2_1\surf\es_R.gii'],
                    [r'H:\workspace\AD_meta\results\meta\1_0\surf\es_L.gii',
                    r'H:\workspace\AD_meta\results\meta\1_0\surf\es_R.gii']]

_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path[0],main_nii_path[1],
                                                        vmax_neg=0, vmin_pos=0,
                                                        cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p[0], p[1], mappable_pos=mappable_pos, mappable_neg=mappable_neg)
# %%
# GMV 未移除协变量
main_nii_path = r'H:\workspace\AD_meta\results\meta\2_0\roi_gmv\es.nii'
sub_nii_pathes = [r'H:\workspace\AD_meta\results\meta\2_1\roi_gmv\es.nii',
                   r'H:\workspace\AD_meta\results\meta\1_0\roi_gmv\es.nii']

_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path, vmax_neg=0,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
# %%
# CT 未移除协变量
main_nii_path = r'H:\workspace\AD_meta\results\meta\2_0\roi_ct\es.nii'
sub_nii_pathes = [r'H:\workspace\AD_meta\results\meta\2_1\roi_ct\es.nii',
                   r'H:\workspace\AD_meta\results\meta\1_0\roi_ct\es.nii']

_, mappable_pos, mappable_neg = mapping_cii.ct_to_cii(main_nii_path, vmax_neg=0,
                                                 vmin_pos=0, vmax_pos=0,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.ct_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
# %%
# 单中心 MCAD GMV
main_nii_path = r'H:\workspace\AD_meta\results\supp\mcad\2_0\roi_gmv_removed\es.nii'
sub_nii_pathes = [r'H:\workspace\AD_meta\results\supp\mcad\2_1\roi_gmv_removed\es.nii',
                   r'H:\workspace\AD_meta\results\supp\mcad\1_0\roi_gmv_removed\es.nii']

_ = mapping_cii.gmv_to_cii(main_nii_path, 
                            cmap_pos=cmap_pos, cmap_neg=cmap_neg)


_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path, 
                                                    vmin_neg=-211, vmax_neg=0,
                                                    vmin_pos=0, vmax_pos=7,
                                                    cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
# %%
# 单中心 MCAD CT
main_nii_path = r'H:\workspace\AD_meta\results\supp\mcad\2_0\roi_ct_removed\es.nii'
sub_nii_pathes = [r'H:\workspace\AD_meta\results\supp\mcad\2_1\roi_ct_removed\es.nii',
                   r'H:\workspace\AD_meta\results\supp\mcad\1_0\roi_ct_removed\es.nii']

_, mappable_pos, mappable_neg = mapping_cii.ct_to_cii(main_nii_path,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

_, mappable_pos, mappable_neg = mapping_cii.ct_to_cii(main_nii_path,
                                                    vmin_neg=-1.39, vmax_neg=0,
                                                    vmin_pos=0, vmax_pos=0,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.ct_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)

#%%
# 单中心 ADNI GMV
main_nii_path = r'H:\workspace\AD_meta\results\supp\adni\2_0\roi_gmv_removed\es.nii'
sub_nii_pathes = [r'H:\workspace\AD_meta\results\supp\adni\2_1\roi_gmv_removed\es.nii',
                   r'H:\workspace\AD_meta\results\supp\adni\1_0\roi_gmv_removed\es.nii']

_ = mapping_cii.gmv_to_cii(main_nii_path, 
                            cmap_pos=cmap_pos, cmap_neg=cmap_neg)

_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path, 
                                                    vmin_neg=-244, vmax_neg=0,
                                                    vmin_pos=0, vmax_pos=18,
                                                    cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
# %%
# 单中心 ADNI CT
main_nii_path = r'H:\workspace\AD_meta\results\supp\adni\2_0\roi_ct_removed\es.nii'
sub_nii_pathes = [r'H:\workspace\AD_meta\results\supp\adni\2_1\roi_ct_removed\es.nii',
                   r'H:\workspace\AD_meta\results\supp\adni\1_0\roi_ct_removed\es.nii']

_, mappable_pos, mappable_neg = mapping_cii.ct_to_cii(main_nii_path,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

_, mappable_pos, mappable_neg = mapping_cii.ct_to_cii(main_nii_path,
                                                    vmin_neg=-1.54, vmax_neg=0,
                                                    vmin_pos=0, vmax_pos=0.1,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.ct_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)

#%%
# 单中心 EDSD GMV
main_nii_path = r'H:\workspace\AD_meta\results\supp\edsd\2_0\roi_gmv_removed\es.nii'
sub_nii_pathes = [r'H:\workspace\AD_meta\results\supp\edsd\2_1\roi_gmv_removed\es.nii',
                   r'H:\workspace\AD_meta\results\supp\edsd\1_0\roi_gmv_removed\es.nii']

_ = mapping_cii.gmv_to_cii(main_nii_path, 
                            cmap_pos=cmap_pos, cmap_neg=cmap_neg)

_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path, 
                                                    vmin_neg=-232, vmax_neg=0,
                                                    vmin_pos=0, vmax_pos=44,
                                                    cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
# %%
# 单中心 EDSD CT
import matplotlib
main_nii_path = r'H:\workspace\AD_meta\results\supp\edsd\2_0\roi_ct_removed\es.nii'
sub_nii_pathes = [r'H:\workspace\AD_meta\results\supp\edsd\2_1\roi_ct_removed\es.nii',
                   r'H:\workspace\AD_meta\results\supp\edsd\1_0\roi_ct_removed\es.nii']

_, mappable_pos, mappable_neg = mapping_cii.ct_to_cii(main_nii_path,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

_, mappable_pos, mappable_neg = mapping_cii.ct_to_cii(main_nii_path,
                                                    vmin_neg=-2.14, vmax_neg=0,
                                                    vmin_pos=0, vmax_pos=0.28,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.ct_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)

#%%
# 其他图谱 SCH
main_nii_path = r'H:\workspace\AD_meta\results\supp\2_0\roi_gmv_schaefer\es.nii'
sub_nii_pathes = [r'H:\workspace\AD_meta\results\supp\2_1\roi_gmv_schaefer\es.nii',
                   r'H:\workspace\AD_meta\results\supp\1_0\roi_gmv_schaefer\es.nii']

_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path,
                                                    vmin_neg=-189.9, vmax_neg=0,
                                                    vmin_pos=0, vmax_pos=34.2,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
# %%
# SCH CT
main_nii_path = r'H:\workspace\AD_meta\results\supp\2_0\schaefer_ct\es.nii'
sub_nii_pathes = [r'H:\workspace\AD_meta\results\supp\2_1\schaefer_ct\es.nii',
                   r'H:\workspace\AD_meta\results\supp\1_0\schaefer_ct\es.nii']

_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path, vmax_neg=0,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path,
                                                    vmin_neg=-219, vmax_neg=0,
                                                    vmin_pos=0, vmax_pos=32,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
#%%
# 其他图谱 AAL
main_nii_path = r'H:\workspace\AD_meta\results\supp\2_0\aal\es.nii'
sub_nii_pathes = [r'H:\workspace\AD_meta\results\supp\2_1\aal\es.nii',
                   r'H:\workspace\AD_meta\results\supp\1_0\aal\es.nii']
_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path, vmax_neg=0,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path,
                                                 vmin_neg=-221.7, vmax_neg=0,
                                                 vmin_pos=0, vmax_pos=10.4,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
# %%
# AAL CT
main_nii_path = r'H:\workspace\AD_meta\results\supp\2_0\aal_ct\es.nii'
sub_nii_pathes = [r'H:\workspace\AD_meta\results\supp\2_1\aal_ct\es.nii',
                   r'H:\workspace\AD_meta\results\supp\1_0\aal_ct\es.nii']
_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path, vmax_neg=0,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)


_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path,
                                                   vmin_neg=-237, vmax_neg=0,
                                                    vmin_pos=0, vmax_pos=21,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)

#%%
import numpy as np
import mapping_cii
import seaborn as sns
from matplotlib import cm, colors
import matplotlib.pyplot as plt

colors1 = cm.gnuplot2(np.linspace(0, 1, 128))[50:]
cmap_neg = colors.LinearSegmentedColormap.from_list('colormap_neg', colors1)
tmp_cm = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)
colors2 = tmp_cm(np.linspace(0, 1, 15))
cmap_pos = colors.LinearSegmentedColormap.from_list('colormap_pos', colors2)
"""
colors2 = cm.BuGn(np.linspace(0, 1, 128))
cmap_pos = colors.LinearSegmentedColormap.from_list('colormap_pos', colors2)
"""
# combine them and build a new colormap
cs = np.vstack((colors1, colors2))
cmap = colors.LinearSegmentedColormap.from_list('my_colormap', cs)
# 绘制colormap
gradient = np.linspace(0, 1, 128)
gradient = np.vstack((gradient, gradient))
plt.imshow(gradient, aspect='auto', cmap=cmap)

#%%
# GMV 亚型4类
main_nii_path = r'./results_0401\subtype\g_agt4\subtype0.nii'
sub_nii_pathes = [r'./results_0401\subtype\g_agt4\subtype1.nii',
                   r'./results_0401\subtype\g_agt4\subtype2.nii',
                   r'./results_0401\subtype\g_agt4\subtype3.nii']

v_min, v_max = mapping_cii.get_min_max_all([main_nii_path]+sub_nii_pathes)
print(v_min, v_max)
#%%
_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path,
                                                 vmax_neg=0, vmin_neg=v_min*100,
                                                 vmax_pos=v_max*100, vmin_pos=0, 
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
#%%
# GMV 亚型3类
main_nii_path = r'./results_0401\subtype\g_agt3\subtype0.nii'
sub_nii_pathes = [r'./results_0401\subtype\g_agt3\subtype1.nii',
                   r'./results_0401\subtype\g_agt3\subtype2.nii']

v_min, v_max = mapping_cii.get_min_max_all([main_nii_path]+sub_nii_pathes)
print(v_min, v_max)
#%%
_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path,
                                                 vmax_neg=0, vmin_neg=v_min*100,
                                                 vmax_pos=v_max*100, vmin_pos=0, 
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
#%%
# GMV 亚型2类
main_nii_path = r'./results_0401\subtype\g_agt2\subtype0.nii'
sub_nii_pathes = [r'./results_0401\subtype\g_agt2\subtype1.nii',]

v_min, v_max = mapping_cii.get_min_max_all([main_nii_path]+sub_nii_pathes)
print(v_min, v_max)
#%%
_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path,
                                                 vmax_neg=0, vmin_neg=v_min*100,
                                                 vmax_pos=v_max*100, vmin_pos=0, 
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)

#%%
# CT 亚型2类
main_nii_path = r'./results_0401\subtype\c_ag2\subtype0.nii'
sub_nii_pathes = [r'./results_0401\subtype\c_ag2\subtype1.nii',]

v_min, v_max = mapping_cii.get_min_max_all([main_nii_path]+sub_nii_pathes)
print(v_min, v_max)
_, mappable_pos, mappable_neg = mapping_cii.ct_to_cii(main_nii_path,
                                                 vmax_neg=0, vmin_neg=v_min,
                                                 vmax_pos=v_max, vmin_pos=0, 
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.ct_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)

#%%
main_nii_path = r'./results/subtype/gmv_ADMCI_4/subtype0.nii'
sub_nii_pathes = [r'./results/subtype/gmv_ADMCI_4/subtype1.nii',
                   r'./results/subtype/gmv_ADMCI_4/subtype2.nii',
                   r'./results/subtype/gmv_ADMCI_4/subtype3.nii']

mapping_cii.gmv_to_cii(main_nii_path, cmap_pos=cmap_pos, cmap_neg=cmap_neg)

_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path,
                                                 vmax_neg=0, vmin_neg=-4746,
                                                 vmax_pos=1411, vmin_pos=0, 
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
# %%
# 不对称性 GMV
# 因为代表的意义不同，使用另一套色谱
cmap_neg = cm.get_cmap('GnBu_r')
cmap_pos = cm.get_cmap('YlOrRd')

main_nii_path = r'./results/supp/asym/2/roi_gmv_removed/es_bon001_top100.nii'
sub_nii_pathes = [r'./results/supp/asym/1/roi_gmv_removed/es_bon001_top100.nii',
                  r'./results/supp/asym/0/roi_gmv_removed/es_bon001_top100.nii']

mapping_cii.gmv_to_cii(main_nii_path, vmax_neg=0,
                        cmap_pos=cmap_pos, cmap_neg=cmap_neg)

_, mappable_pos, mappable_neg = mapping_cii.gmv_to_cii(main_nii_path, 
                                                 vmin_neg=-553.76, vmax_neg=0, 
                                                 vmin_pos=0, vmax_pos=521.2, 
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.gmv_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
# %%
# 不对称性 CT
# 因为代表的意义不同，使用另一套色谱

main_nii_path = r'./results/supp/asym/2/roi_ct_removed/es_bon001_top100.nii'
sub_nii_pathes = [r'./results/supp/asym/1/roi_ct_removed/es_bon001_top100.nii',
                  r'./results/supp/asym/0/roi_ct_removed/es_bon001_top100.nii']

mapping_cii.ct_to_cii(main_nii_path,
                        cmap_pos=cmap_pos, cmap_neg=cmap_neg)

_, mappable_pos, mappable_neg = mapping_cii.ct_to_cii(main_nii_path,
                                                 vmax_neg=0, vmin_neg=0,
                                                 vmax_pos=2.987, vmin_pos=0,
                                                 cmap_pos=cmap_pos, cmap_neg=cmap_neg)

for p in sub_nii_pathes:
    mapping_cii.ct_to_cii(p, mappable_pos=mappable_pos, mappable_neg=mappable_neg)
# %%
