#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from matplotlib.axes import Axes
from matplotlib.projections.polar import PolarAxes
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.markers import MarkerStyle

import matplotlib.colors

import seaborn as sns

rgbs = ['#fd8d3c', '#74c476', '#6baed6']
custom_cmap = matplotlib.colors.ListedColormap(rgbs, name='my_colormap')
plt.ioff()

def models_to_dataframe(models):
    index = models.keys()
    cols = ['es', 'll', 'ul', 'p']
    df = pd.DataFrame(index=index, columns=cols)
    for k,v in models.items():
        df.loc[k]['es'] = v.total_effect_size
        df.loc[k]['ll'] = v.total_lower_limit
        df.loc[k]['ul'] = v.total_upper_limit
        df.loc[k]['p'] = v.p
    return df

def draw_top(main_models, sub_models_list, cmap=custom_cmap,
             cmap_start=0, cmap_step=4,
             legend_names=None, topn=20, offset=0.2,
             width_ratio=0.1, height_ratio=0.2,
             linewidth=1, point_size=5, fontsize=12,
             box_aspect=None, value_aspect='auto',
             id_csv_path='./data/mask/cortical_id_new.csv',
             show=True, out_path=None):
    # load id Dataframe
    id_df = pd.read_csv(id_csv_path, index_col='id')

    main_df = models_to_dataframe(main_models)
    sub_dfs  = [models_to_dataframe(models) for models in sub_models_list]

    sorted_main_df = main_df.sort_values('es')
    top_df = sorted_main_df[:topn]
    
    colors = cmap(np.arange(cmap_start, 1+cmap_start+len(sub_models_list))*cmap_step, cmap_step).tolist()
    main_color = colors.pop(0)

    fig = plt.figure(figsize=(width_ratio*topn, height_ratio*topn))
    ax = fig.add_axes([0, 0, 1, 1])
    legends = []

    y_labels =[]
    y = topn
    for index, row in top_df.iterrows():
        ll = row['ll']
        ul = row['ul']
        es = row['es']

        plt.scatter(es, y, s=point_size, color=main_color)
        b, = plt.plot((ll, ul), (y, y), linewidth=linewidth, color=main_color)
        if y == topn:
            legends.append(b)
        y = y - 1

        y_labels.append(id_df.loc[int(index)]['name'])

    i = 1
    offsets = []
    for i in range(2,len(sub_models_list)+2):
        if i % 2:
            offsets.append(offset*int(i/2))
        else:
            offsets.append(-offset*int(i/2))
    top_sub_dfs = [top_df.align(df, join='left')[1] for df in sub_dfs]
    for sub_df, color, _offset in zip(top_sub_dfs, colors, offsets):
        y = topn
        for index, row in sub_df.iterrows():
            ll = row['ll']
            ul = row['ul']
            es = row['es']

            plt.scatter(es, y+_offset, s=point_size, color=color)
            b, = plt.plot((ll, ul), (y+_offset, y+_offset), linewidth=linewidth, color=color)
            if y == topn:
                legends.append(b)
            y = y - 1
        i += 1
    
    ax.set_yticks(np.arange(topn+1))
    y_labels.append('')
    y_labels.reverse()
    ax.set_yticklabels(y_labels, fontdict={'fontsize':fontsize})
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='y', length=0)
    if legend_names is not None and len(legend_names)==len(legends):
        plt.legend(legends, legend_names)
    ax.set_box_aspect(aspect=box_aspect)
    ax.set_aspect(aspect=value_aspect)

    if out_path is not None:
        plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()

# Draw cesa radar plot
def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=False, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, **kwargs):
            self.set_thetagrids(np.degrees(theta), labels, rotation=45,**kwargs)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def radar_plot(dfs, col_name, p_thres=0.05, cmap=custom_cmap,
               cmap_start=0, cmap_step=4,
               legend_loc=(-0.2,-0.2), legend_names=None,
               out_path=None, show=True, save=False):
    n = len(dfs[0])
    theta = radar_factory(n, frame='circle')
    _, ax = plt.subplots(subplot_kw=dict(projection='radar'))

    colors = cmap(np.arange(cmap_start, 1+cmap_start+len(dfs)*cmap_step, cmap_step)).tolist()

    labels = [0 for i in range(n)]
    for df,color in zip(dfs, colors):
        values = df[col_name]
        i = 0
        # mark significant label
        for value in values:
            if value < p_thres:
                labels[i] = 1
            i+=1
        ax.plot(theta, -np.log10(values), color=color)
    spoke_labels = list(dfs[0].index)

    i = 0
    for label in labels:
        if not label:
            spoke_labels[i] = ""
        i += 1

    if legend_names is not None:
        ax.legend(legend_names, loc=legend_loc)

    # Rotate labels
    ax.set_xticks(theta)
    ax.set_xticklabels(spoke_labels)

    angles = np.linspace(0,2*np.pi,len(ax.get_xticklabels())+1)
    angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
    angles = np.rad2deg(angles)

    for label, angle in zip(ax.get_xticklabels(), angles):
        x,y = label.get_position()
        lab = ax.text(x,y-0.2, label.get_text(), transform=label.get_transform(),
                    ha=label.get_ha(), va=label.get_va())
        if angle % 180 > 90:
            lab.set_rotation(angle+90)
        else:
            lab.set_rotation(angle-90)
        labels.append(lab)
    ax.set_xticklabels([])
    # inverse y axis for display
    ax.invert_yaxis()
    if save:
        plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_pet_results(result_dict, show=True, save=True,
                     fig_width=10, fig_height=5,
                     cmap=custom_cmap,
                     alpha=0.7, fontsize=12, p=0.05,
                     out_path='./results/correlation/PET.png'):
    gmv_marker = MarkerStyle(marker='o')
    ct_marker = MarkerStyle(marker='^')

    _, ax = plt.subplots(figsize=(float(fig_width), float(fig_height)))
    legend = []
    labels = []

    colors = cmap(np.arange(len(result_dict)))

    for k, v in result_dict.items():
        name = []
        if 'AD_NC' in k:
            c = colors[0]
            offset = -0.2
        elif 'AD_MCI' in k:
            c = colors[1]
            offset = 0
        elif 'MCI_NC' in k:
            c = colors[2]
            offset = 0.2
        if 'GMV' in k:
            marker = gmv_marker
        elif 'CT' in k:
            marker = ct_marker
        sign_x = []
        sign_y = []
        not_sign_x = []
        not_sign_y = []
        x = 1
        for result in v:
            if result.p < p:
                sign_x.append(x)
                sign_y.append(result.r)
            else:
                not_sign_x.append(x)
                not_sign_y.append(result.r)
            x += 1
            name.append(result.name)
        sign_x = np.array(sign_x)
        sign_x = sign_x + offset
        ls = ax.scatter(sign_x, sign_y, color=c, alpha=alpha, marker=marker)
        #ax.scatter(not_sign_x, not_sign_y, color=c, facecolors='none', alpha=alpha, marker=marker)
        labels.append(k)
        legend.append(ls)
    ax.axhline(0, color='black', lw=1)
    #ax.set_ylim(-1, 1)
    """
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(legend, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    """
    ax.legend(legend, labels, prop={'size': 8}, loc=(1.1, 0.2))
    
    ax.set_xticks(range(1, len(name)+1))
    ax.set_xticklabels(name)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    
    if save:
        plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()

#%%
def get_ages_and_gmv(young_centers, old_centers, label=2, roi=1):
    all_ages = []
    all_roi_values = []
    for center in young_centers:
        roi_values, *_ = center.get_csv_values(label=0,
                            prefix='roi_gmv/{}.csv',
                            flatten=True)
        roi_values = roi_values[:, roi-1]
        ages, _ = center.get_ages(label=0)
        all_roi_values.append(roi_values)
        all_ages.append(ages)
    for center in old_centers:
        roi_values, *_ = center.get_csv_values(label=label,
                            prefix='roi_gmv/{}.csv',
                            flatten=True)
        if roi_values is not None:
            roi_values = roi_values[:, roi-1]
            ages, _ = center.get_ages(label=label)
            all_roi_values.append(roi_values)
            all_ages.append(ages*100)
    all_roi_values = np.concatenate(all_roi_values)
    all_ages = np.concatenate(all_ages)
    return all_ages, all_roi_values

def plot_roi_aging(young_centers, old_centers, labels, roi):
    slabels = ['NC', 'MCI', 'AD']
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5), sharex=True, sharey=True)
    all_ages, all_roi_values = get_ages_and_gmv(young_centers, old_centers, label=labels[0], roi=roi)
    ax1.scatter(all_ages, all_roi_values, alpha=0.5)
    ax1.set_title(slabels[labels[0]]+'-'+str(roi))

    all_ages, all_roi_values = get_ages_and_gmv(young_centers, old_centers, label=labels[1], roi=roi)
    ax2.scatter(all_ages, all_roi_values, alpha=0.5)
    ax2.set_title(slabels[labels[1]]+'-'+str(roi))
# %%
def plot_mmse_cor(centers, roi=1):
    all_ages = []
    all_roi_values = []
    for center in centers:
        roi_values, *_ = center.get_csv_values(
                            prefix='roi_gmv/{}.csv',
                            flatten=True)
        if roi_values is not None:
            roi_values = roi_values[:, roi-1]
            ages, _ = center.get_MMSEs()
            all_roi_values.append(roi_values)
            all_ages.append(ages)
    all_roi_values = np.concatenate(all_roi_values)
    all_ages = np.concatenate(all_ages)

    r = pearsonr(all_ages, all_roi_values)[0]
    p = pearsonr(all_ages, all_roi_values)[1]
    plt.scatter(all_ages, all_roi_values, alpha=0.5)
    plt.title('r:{:.2f}, p:{:.2e}'.format(r, p))
    plt.show()

def plot_correlation(values1, values2, x_label='x', y_label='y',
                     out_path=None, show=True, save=False,
                     fig_width=5, fig_height=5, fontsize=14):
    r = pearsonr(values1, values2)[0]
    p = pearsonr(values1, values2)[1]

    df = pd.DataFrame(
                    {x_label: values1,
                    y_label: values2,
                    })

    _, ax = plt.subplots(figsize=(float(fig_width), float(fig_height)))
    ax = sns.regplot(x=x_label, y=y_label, data=df, robust=True,
                     ax=ax)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_title('r={:.2f}, p={:.2e}'.format(r, p), fontdict={'fontsize': fontsize})
    if save:
        plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()

def plot_correlation_joint(values1, values2, x_label='x', y_label='y',
                        out_path=None, show=True, save=False,
                        fig_width=5, fig_height=5, fontsize=14):
    r, p = pearsonr(values1, values2)
    df = pd.DataFrame(
                    {x_label: values1,
                    y_label: values2,
                    })

    g = sns.jointplot(x=x_label, y=y_label, data=df, kind="hex")
    sns.regplot(x=x_label, y=y_label, data=df, scatter=False,
                ci=None, ax=g.ax_joint)
    g.ax_marg_x.remove()
    g.ax_marg_y.remove()

    g.ax_joint.tick_params(axis='both', which='major', labelsize=fontsize)
    g.set_axis_labels(x_label, y_label, fontsize=fontsize)
    plt.title('r={:.2f}, p={:.2e}'.format(r, p), 
              fontdict={'fontsize': fontsize})
    if save:
        g.savefig(out_path)
    if show:
        plt.show()
    plt.close()
    
def plot_correlation_joint_dist(values1, values2, x_label='x', y_label='y',
                        out_path=None, show=True, save=False,
                        fig_width=5, fig_height=5, fontsize=14):
    r, p = pearsonr(values1, values2)
    df = pd.DataFrame(
                    {x_label: values1,
                    y_label: values2,
                    })

    g = sns.jointplot(x=x_label, y=y_label, data=df, kind="hex")

    g.ax_joint.tick_params(axis='both', which='major', labelsize=fontsize)
    g.set_axis_labels(x_label, y_label, fontsize=fontsize)
    plt.title('r={:.2f}, p={:.2e}'.format(r, p), 
              fontdict={'fontsize': fontsize})
    if save:
        g.savefig(out_path)
    if show:
        plt.show()
    plt.close()
    
def draw_stem(gmv_csv_path=r'E:\workspace\tesa\results\CESA\GMV.csv',
             ct_csv_path=r'E:\workspace\tesa\results\CESA\CT.csv',
             index_col=0,
             col='0.05 - adjusted',
             fontsize=18, alpha=0.6,
             fig_width=3, fig_height=6):
    df = pd.read_csv(gmv_csv_path, index_col=index_col)
    df = df.sort_values(col, ascending=False)

    sub_df = pd.read_csv(ct_csv_path, index_col=index_col)
    sub_df = sub_df.loc[df.index]

    y = range(0, len(df.index))
    values = -np.log10(df[col].to_numpy())
    sub_values = -np.log10(sub_df[col].to_numpy())
    y_labels = df.index.to_numpy().tolist()

    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.scatter(x=values, y=y, label='GMV', alpha=alpha)
    ax.hlines(y=y, xmin=0, xmax=values)
    ax.scatter(x=sub_values, y=y, label='CT', alpha=alpha)
    ax.hlines(y=y, xmin=0, xmax=sub_values, color='C1')

    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, {'fontsize': 10}, ha='right')
    ax.set_xlabel('-log10 adjusted p-value', fontsize=fontsize)
    ax.set_ylabel('Gene Sets', fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.draw()

    ax.legend(loc="lower right")
    plt.show()
    plt.close()

def draw_fuma(csv_path='./results/gene/2_0/FUMA_gmv_pos_150/gtex_v8_ts_DEG.csv',
              sub_csv_path='./results/gene/2_0/FUMA_ct_neg_150/gtex_v8_ts_DEG.csv',
              cate='DEG.up', topn=10,
              fontsize=18, alpha=0.6,
              fig_width=4, fig_height=6):
    df = pd.read_csv(csv_path, index_col=1)
    df = df[df['Category']==cate]
    df = df.sort_values('adjP', ascending=True)
    df = df[:topn]
    df = df.sort_values('adjP', ascending=False)

    sub_df = pd.read_csv(sub_csv_path, index_col=1)
    sub_df = sub_df[sub_df['Category']==cate]
    sub_df = sub_df.loc[df.index]

    y = range(0, topn)
    values = -np.log10(df['adjP'].to_numpy())
    sub_values = -np.log10(sub_df['adjP'].to_numpy())
    y_labels = df.index.to_numpy().tolist()

    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.barh(y, width=values, label='GMV', alpha=alpha)
    ax.barh(y, width=sub_values, label='CT', alpha=alpha)

    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, {'fontsize': 12, 'weight': 'bold'}, ha='left')
    ax.set_xlabel('-log10 p-value', fontsize=fontsize)
    ax.set_ylabel('Gene Set', fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.draw()

    yax = ax.get_yaxis()
    pad = min(T.label.get_window_extent().width for T in yax.majorTicks)
    yax.set_tick_params(pad=-5)

    ax.legend()
    plt.show()
    plt.close()

def draw_fuma_stem(csv_path='./results/gene/2_0/FUMA_gmv_pos_150/gtex_v8_ts_DEG.csv',
                sub_csv_path='./results/gene/2_0/FUMA_ct_neg_150/gtex_v8_ts_DEG.csv',
                cate='DEG.up', 
                fontsize=18, alpha=0.6,
                fig_width=3, fig_height=10):
    df = pd.read_csv(csv_path, index_col=1)
    df = df[df['Category']==cate]
    df = df.sort_values('adjP', ascending=False)

    sub_df = pd.read_csv(sub_csv_path, index_col=1)
    sub_df = sub_df[sub_df['Category']==cate]
    sub_df = sub_df.loc[df.index]

    y = range(0, len(df.index))
    values = -np.log10(df['adjP'].to_numpy())
    sub_values = -np.log10(sub_df['adjP'].to_numpy())
    y_labels = df.index.to_numpy().tolist()

    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.scatter(x=values, y=y, label='GMV', alpha=alpha)
    ax.hlines(y=y, xmin=0, xmax=values)
    ax.scatter(x=sub_values, y=y, label='CT', alpha=alpha)
    ax.hlines(y=y, xmin=0, xmax=sub_values, color='C1')

    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, {'fontsize': 10}, ha='right')
    ax.set_xlabel('-log10 FDR q-value', fontsize=fontsize)
    ax.set_ylabel('Gene Set', fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.draw()

    ax.legend(loc="lower right")
    plt.show()
    plt.close()

def draw_fuma_stem_yx(csv_path='./results/gene/2_0/FUMA_gmv_pos_150/gtex_v8_ts_DEG.csv',
                sub_csv_path='./results/gene/2_0/FUMA_ct_neg_150/gtex_v8_ts_DEG.csv',
                cate='DEG.up', 
                fontsize=18, alpha=0.6,
                fig_width=10, fig_height=2):
    df = pd.read_csv(csv_path, index_col=1)
    df = df[df['Category']==cate]
    df = df.sort_values('adjP', ascending=True)

    sub_df = pd.read_csv(sub_csv_path, index_col=1)
    sub_df = sub_df[sub_df['Category']==cate]
    sub_df = sub_df.loc[df.index]

    x = range(0, len(df.index))
    values = -np.log10(df['adjP'].to_numpy())
    sub_values = -np.log10(sub_df['adjP'].to_numpy())
    x_labels = df.index.to_numpy().tolist()

    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.scatter(y=values, x=x, label='GMV', alpha=alpha)
    ax.vlines(x=x, ymin=0, ymax=values)
    ax.scatter(y=sub_values, x=x, label='CT', alpha=alpha)
    ax.vlines(x=x, ymin=0, ymax=sub_values, color='C1')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, {'fontsize': 10}, ha='right', rotation=90)
    ax.set_ylabel('-log10 p-value', fontsize=fontsize)
    ax.set_xlabel('Gene Set', fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.draw()

    ax.legend(loc="lower right")
    plt.show()
    plt.close()

def draw_pet_relplot(df, x_labels, y_labels):
    # Draw each cell as a scatter point with varying size and color
    sns.set_theme(style="whitegrid")
 
    g = sns.relplot(
        data=df,
        x="x", y="y", hue="r", size="absr",
        palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
        height=10, sizes=(50, 250), size_norm=(-.2, .8),)

    g.ax.set_xticks(np.arange(1, len(x_labels)+1))
    g.ax.set_xticklabels(x_labels)

    g.ax.set_yticks(np.arange(1, len(y_labels)+1))
    g.ax.set_yticklabels(y_labels)
    
    g.despine(left=True, bottom=True)
    g.ax.margins(.02)

    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    for artist in g.legend.legendHandles:
        artist.set_edgecolor(".7")

    plt.show()