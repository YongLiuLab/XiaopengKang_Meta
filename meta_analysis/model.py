""" model module as meta analysis model

Function:
    inverse_variance(variance): inverse variance
    get_confidence_intervals(effect_size, standard_error): caculate 95% confidence intervals
    get_heterogeneity(effect_sizes, total_effect_size, weights)： caculate heterogeneity
    get_z_value(total_effect_size, standard_error, x0): caculate z test value
    get_p_from_z(z, one_side): caculate p value from z value

Class:
    Model(object): basic model
    FixedModel(Model): Fixed model, assume one true effect size that underlies all the studies.
    RandomModel(Model): Random model, assume effect size varies from study to study.

Author: Kang Xiaopeng
Data: 2020/02/21
E-mail: kangxiaopeng2018@ia.ac.cn

This file is part of meta_analysis.

meta_analysis is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

meta_analysis is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with meta_analysis.  If not, see <https://www.gnu.org/licenses/>.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

import numpy as np
from scipy.stats import norm

from . import data

def inverse_variance(variance):
    return 1 / variance

def get_confidence_intervals(effect_size, standard_error):
    # 95% confidence intervals
    lower_limits = effect_size - 1.96 * standard_error
    upper_limits = effect_size + 1.96 * standard_error
    return lower_limits, upper_limits

def get_heterogeneity(effect_sizes, total_effect_size, weights):
    diff_es_square = (effect_sizes - total_effect_size) ** 2
    q = np.sum(np.multiply(weights, diff_es_square))
    return q

def get_z_value(total_effect_size, standard_error, x0=0):
    return (total_effect_size - x0) / standard_error

def get_p_from_z(z, one_side=False):
    if one_side:
        p_value = norm.sf(abs(z))
    else:
        p_value = norm.sf(abs(z)) * 2
    return p_value

class Model(object):
    """Basic class to perform check effect size from all study.

    Attributes:
        studies: list of Study instance.
        effect_sizes: list, all studies' effect size
        variances: list, all studies' variance
        weights: list, all studies' weight
        total_effect_size: float, combined effect size
        total_variance: float, combined variance 
        standard_error: float, standard error of meta analysis
        lower_limits: float, 95% lower confidence intervals
        upper_limits: float, 95% upper confidence intervals
        q: float, heterogeneity
        z: float, z test value
        p: float, p value

    Function:
        gen_weights(): caculate weight from effect_sizes and variances
        caculate(): caculate results
        get_results(): return all results.
        plot_forest(): show forest plot.
    """

    def __init__(self, studies):
        super().__init__()
        self.studies = studies
        effect_sizes = []
        variances = []
        lower_limits = []
        upper_limits = []
        for study in studies:
            es = study.get_effect_size()
            v = study.get_variance()
            ll, ul = study.get_confidence_intervals()
            effect_sizes.append(es)
            variances.append(v)
            lower_limits.append(ll)
            upper_limits.append(ul)

        self.effect_sizes = np.asarray(effect_sizes)
        self.variances = np.asarray(variances)
        self.lower_limits = np.asarray(lower_limits)
        self.upper_limits = np.asarray(upper_limits)
        self.gen_weights()

        self.caculate()
    
    def gen_weights(self):
        raise NotImplementedError("Generate Weight Method Not Implemented")

    def caculate(self):
        effect_sizes = self.effect_sizes
        variances = self.variances
        weights = self.weights

        total_effect_size = np.sum(np.multiply(effect_sizes, weights)) /\
                   np.sum(weights)
        total_variance = 1 / np.sum(weights)
        total_standard_error = np.sqrt(total_variance)

        total_lower_limit, total_upper_limit = get_confidence_intervals(total_effect_size, total_standard_error)
        q = get_heterogeneity(effect_sizes, total_effect_size, weights)
        z = get_z_value(total_effect_size, total_standard_error)
        p = get_p_from_z(z)

        self.total_effect_size = total_effect_size
        self.total_variance = total_variance
        self.total_standard_error = total_standard_error
        self.total_lower_limit = total_lower_limit
        self.total_upper_limit = total_upper_limit
        self.q = q
        self.z = z
        self.p = p

    def get_results(self):
        return (self.total_effect_size,
                self.total_variance, self.total_standard_error,
                self.total_lower_limit, self.total_upper_limit,
                self.q, self.z, self.p)

    def plot_forest(self, title='meta analysis', 
                    plot_group_details=True,save_path=None,
                    show=True, font_size=18,
                    grid_width=1, grid_height=0.5,
                    every_nth=3):
        plt.ioff()
        dpi = 100
        forest_plot_width = 4 * grid_width
        forest_plot_height = (len(self.studies) + 1) * grid_height

        is_cont = True
        data_type = self.studies[0].data_type
        if data_type == data.Study.num:
            is_cont = True
        elif data == data.Study.cate:
            is_cont = False
        
        if not is_cont:
            plot_group_details = False

        width = 18
        if not plot_group_details:
            width = width - 6

        height = 2 * grid_height + forest_plot_height

        fig = plt.figure(figsize=(width, height),
                               dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        # set default to invisible
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)

        x, y = 0, 0
        llha = 'left'
        blva = 'bottom'
        ha = 'center'
        va = 'top'

        study_x = x
        experimental_group_x = study_x + grid_width * 4
        control_group_x = experimental_group_x + grid_width * 3
        if plot_group_details:
            froest_plot_x = control_group_x + 2 + forest_plot_width / 2 
        else: 
            froest_plot_x = study_x + 2 + forest_plot_width / 2 
        effect_size_x = froest_plot_x + 1 + forest_plot_width / 2
        lower_limit_x = effect_size_x + grid_width
        upper_limit_x = lower_limit_x + grid_width
        interval_x = (lower_limit_x + upper_limit_x) / 2
        weight_x = upper_limit_x + grid_width
        header_y = height

        #draw header
        ax.text(study_x, header_y, 'Study',ha=llha, va=va, fontsize=font_size)
        if plot_group_details:
            ax.text(experimental_group_x, header_y, 'experimental group',ha=ha, va=va, fontsize=font_size)
            ax.text(control_group_x, header_y, 'control group',ha=ha, va=va, fontsize=font_size)
        ax.text(froest_plot_x, header_y, 'forest plot',ha=ha, va=va, fontsize=font_size)
        ax.text(effect_size_x, header_y, 'effect size',ha=ha, va=va, fontsize=font_size)
        ax.text(interval_x, header_y, '95% interval',ha=ha, va=va, fontsize=font_size)
        ax.text(weight_x, header_y, 'weight',ha=ha, va=va, fontsize=font_size)
        #draw subheader
        subheader_y = height - grid_height / 2
        if plot_group_details:
            eg_mean_x = experimental_group_x - grid_width
            eg_std_x = experimental_group_x
            eg_count_x = experimental_group_x + grid_width
            cg_mean_x = control_group_x - grid_width
            cg_std_x = control_group_x
            cg_count_x = control_group_x + grid_width

            ax.text(eg_mean_x, subheader_y, 'mean',ha=ha, va=va, fontsize=font_size)
            ax.text(eg_std_x, subheader_y, 'std',ha=ha, va=va, fontsize=font_size)
            ax.text(eg_count_x, subheader_y, 'count',ha=ha, va=va, fontsize=font_size)
            ax.text(cg_mean_x, subheader_y, 'mean',ha=ha, va=va, fontsize=font_size)
            ax.text(cg_std_x, subheader_y, 'std',ha=ha, va=va, fontsize=font_size)
            ax.text(cg_count_x, subheader_y, 'count',ha=ha, va=va, fontsize=font_size)

        total_eg_count = 0
        total_cg_count = 0

        row_y = height - grid_height * 1.4
        ax.axhline((subheader_y+row_y)/2, color='black')
        # draw Study details

        weights = np.reciprocal(self.variances)
        weights = weights / np.sum(weights)
        first_row_y = height - grid_height * 1.5
        for i, (study, effect_size, weight,
                lower_limit, upper_limit) in enumerate(
                    zip(self.studies, self.effect_sizes, weights,
                        self.lower_limits, self.upper_limits)):
            row_y = first_row_y - grid_height * i
            eg_mean, eg_std, eg_count = study.group_experimental.get_mean_std_count()
            cg_mean, cg_std, cg_count = study.group_control.get_mean_std_count()

            ax.text(study_x, row_y, study.name, ha=llha, va=va, fontsize=font_size)
            if plot_group_details:
                ax.text(eg_mean_x, row_y, '{:.2f}'.format(eg_mean), ha=ha, va=va, fontsize=font_size)
                ax.text(eg_std_x, row_y, '{:.2f}'.format(eg_std), ha=ha, va=va, fontsize=font_size)
                ax.text(eg_count_x, row_y, int(eg_count), ha=ha, va=va, fontsize=font_size)
                ax.text(cg_mean_x, row_y, '{:.2f}'.format(cg_mean), ha=ha, va=va, fontsize=font_size)
                ax.text(cg_std_x, row_y, '{:.2f}'.format(cg_std), ha=ha, va=va, fontsize=font_size)
                ax.text(cg_count_x, row_y, int(cg_count), ha=ha, va=va, fontsize=font_size)
            
            ax.text(effect_size_x, row_y, '{:.2f}'.format(effect_size), ha=ha, va=va, fontsize=font_size)
            ax.text(lower_limit_x, row_y, '[{:.2f}'.format(lower_limit), ha=ha, va=va, fontsize=font_size)
            ax.text(upper_limit_x, row_y, '{:.2f}]'.format(upper_limit), ha=ha, va=va, fontsize=font_size)
            ax.text(weight_x, row_y, '{:.2f}%'.format(weight*100), ha=ha, va=va, fontsize=font_size)

            total_eg_count += eg_count
            total_cg_count += cg_count
        # draw total
        total_y = row_y - grid_height
        ax.text(study_x, total_y, 'Total', ha=llha, va=va, fontsize=font_size)
        if plot_group_details:
            ax.text(eg_count_x, total_y, int(total_eg_count), ha=ha, va=va, fontsize=font_size)
            ax.text(cg_count_x, total_y, int(total_cg_count), ha=ha, va=va, fontsize=font_size)
        ax.text(effect_size_x, total_y, '{:.2f}'.format(self.total_effect_size), ha=ha, va=va, fontsize=font_size)
        ax.text(lower_limit_x, total_y, '[{:.2f}'.format(self.total_lower_limit), ha=ha, va=va, fontsize=font_size)
        ax.text(upper_limit_x, total_y, '{:.2f}]'.format(self.total_upper_limit), ha=ha, va=va, fontsize=font_size)
        ax.text(weight_x, total_y, '{:.2f}%'.format(np.sum(weights)*100), ha=ha, va=va, fontsize=font_size)
        
        # draw summary
        summary_y = -grid_height/2
        ax.axhline(0, 0, 1, color='black')
        ax.text(0, summary_y, 'Heterogeneity:{:.2f}'.format(self.q), ha=llha, va=blva, fontsize=font_size)
        ax.text(effect_size_x, summary_y, 'z-value:{:.2f}'.format(self.z), ha=ha, va=blva, fontsize=font_size)
        ax.text(upper_limit_x, summary_y, 'p-value:{:.2e}'.format(self.p), ha=ha, va=blva, fontsize=font_size)
        ax.text(froest_plot_x, summary_y, 'Title:{}'.format(title), ha=ha, va=blva, fontsize=font_size)
        
        # draw froest plot
        forest_ax = fig.add_axes([(froest_plot_x-forest_plot_width/2)/width, grid_height/height,
                                   forest_plot_width/width, forest_plot_height/height])
        forest_ax.tick_params(left=False, labelleft=False)
        forest_ax.set_ylim(0, forest_plot_height)

        forest_first_row_y = forest_plot_height - grid_height * 0.5
        largest_box_scale = 2 / 5
        largest_box_width = grid_width * largest_box_scale
        largest_box_height = grid_height * largest_box_scale
        
        accent_color = 'grey'

        forest_ax.spines['left'].set_color('none')
        forest_ax.spines['right'].set_color('none')
        forest_ax.spines['top'].set_color('none')

        forest_ax.axvline(0, 0, 1, color='black')

        box_weights = weights / np.max(weights)
        for i, (effect_size, weight, lower_limit, upper_limit) in enumerate(
                zip(self.effect_sizes, box_weights, self.lower_limits, self.upper_limits)):
            
            row_y = forest_first_row_y - grid_height * i

            box_width = weight * largest_box_width
            box_height = largest_box_height
            box_left_x = effect_size - box_width / 2
            box_bottom_y = row_y - box_height / 2

            color = 'black'
            if lower_limit * upper_limit < 0:
                color = accent_color

            forest_ax.plot([lower_limit, upper_limit], [row_y, row_y], color=color)
            #forest_ax.plot([lower_limit, lower_limit], [box_bottom_y, box_bottom_y+box_height], color=color)
            #forest_ax.plot([upper_limit, upper_limit], [box_bottom_y, box_bottom_y+box_height], color=color)
            forest_ax.add_patch(Rectangle(xy=(box_left_x, box_bottom_y),
                                  width=box_width, 
                                  height=box_height,
                                  fc=color,
                                  alpha=0.5))
        for tick in forest_ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for n, label in enumerate(forest_ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)

        if self.q > 0.5:
            fill = True
        else:
            fill = False

        forest_total_y = row_y - grid_height
        color = 'black'
        if self.total_lower_limit * self.total_upper_limit < 0:
            color = accent_color
        diamond = np.asarray([[self.total_lower_limit, forest_total_y],
                              [self.total_effect_size, forest_total_y + largest_box_height/2],
                              [self.total_upper_limit, forest_total_y], 
                              [self.total_effect_size, forest_total_y - largest_box_height/2]])
        forest_ax.add_patch(Polygon(diamond, fill=fill, fc=color))

        if show:
            plt.show()
        if save_path:
            fig.savefig(save_path)

class FixedModel(Model):
    def __init__(self, studies):
        super().__init__(studies)

    def gen_weights(self):
        self.weights = np.reciprocal(self.variances)

class RandomModel(Model):
    def __init__(self, studies):
        super().__init__(studies)

    def gen_weights(self):
        effect_sizes = self.effect_sizes
        variances = self.variances

        fixed_weights = np.reciprocal(variances)
        mean_effect_size = np.dot(effect_sizes, fixed_weights)/np.sum(fixed_weights)
        Q = np.sum(np.square(effect_sizes-mean_effect_size)/variances)
        df = len(variances) - 1
        C = np.sum(fixed_weights) - np.sum(np.square(fixed_weights)) / np.sum(fixed_weights)
        tau_square = (Q - df) / C
        if tau_square < 0:
            tau_square = 0
        self.tau_square = tau_square
        self.weights = np.reciprocal(variances + tau_square)