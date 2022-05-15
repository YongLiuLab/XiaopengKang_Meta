""" data module mainly use to manage core data

Class:
    Group(object): A specific label group within a center
    Study(object): A study which hold two group's mean, std, count,
                   can caculate its effect size, variance.
    Center(object): A center holds lots of group, generate study.
    Cneters(object): Hold list of Center instances.

Author: Kang Xiaopeng
Data: 2020/02/20
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

from dataclasses import dataclass
from typing import Any
import math

import nibabel as nib
import numpy as np

from . import utils

class Group(object):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def get_label(self):
        return self.label

class CategoricalGroup(Group):
    def __init__(self, label, exposed, not_exposed):
        super().__init__(label)
        self.exposed = exposed
        self.not_exposed = not_exposed

    def get_values(self):
        return self.exposed, self.not_exposed

class NumericalGroup(Group):
    """ Group of specific label, holds origin data or mean, std, count.

    Attributes:
        label: int or string, this group's label.
        datas: original datas, 1d array, used to caculate mean, std, count.
        mean: float, mean of datas
        std: float, std of datas
        count: int, count of datas
        shape: default 1, used to check mean's dimension.
        not_msn: bool, whether one of (mean, std, count) is None

    Function:
        check_property(): check if not_msn and not datas
        is_same_shape(): check if other group has same shape with self
        get_label(): return label
        get_mean_std_count(): return mean, std, count
    """

    def __init__(self, label, datas=None,
                 mean=None, std=None, count=None):
        """ init, Note must input (mean, std, count) or datas
        Args:
            label: int or string, this group's label.
            datas: original datas, 1d array, used to caculate mean, std, count.
            mean: float, mean of datas
            std: float, std of datas
            count: float, count of datas
        """
        super().__init__(label)
        self.datas = datas
        self.mean = mean
        self.std = std
        self.count = count

    def check_property(self):
        self.not_msn = self.mean is None or self.std is None or not self.count
        if self.not_msn:
            if self.datas:
                return True
            elif not self.datas:
                raise AttributeError('Need input for datas or (mean, std, count)')
        else:
            return True
    
    def get_mean_std_count(self):
        if self.check_property():
            if self.not_msn:
                self.mean, self.std, self.count = utils.cal_mean_std_n(np.asarray(self.datas))
                self.not_msn = False
        return self.mean, self.std, self.count

@dataclass
class Study(object):
    """ meta analysis study, used to caculate effect size and variance.

    Attributes:
        name: str, study name
        method: str, method to caculate effect size 
        group_experimental: Group, experimental group
        group_control, Group, control group
        effect_size: float, caculated effect size using 'method'.
        variance: float, variacnce.
        standard_error: float, standard error
        
    Function:
        cohen_d(): cohen's d effect size
        hedge_g(): hedge's g effect size
        get_effect_size(): use specific method to return effect size
        get_variance(): return variance
    """
    name: str
    method: str
    group_experimental: Group
    group_control: Group
    num = 1
    cate = 0

    def __post_init__(self):
        method = self.method.lower()
        if method == 'cohen_d' or method == 'cohen':
            func = self.cohen_d
            data_type = self.num
        elif method == 'hedge_g' or method == 'hedge':
            func = self.hedge_g
            data_type = self.num
        elif method == 'risk_ratio' or method == 'rr':
            func = self.risk_ratio
            data_type = self.cate
        self.data_type = data_type
        self.func = func
        self.func()

    def cohen_d(self):
        """ details in https://en.wikipedia.org/wiki/Effect_size
        """
        m1, s1, n1 = self.group_experimental.get_mean_std_count()
        m2, s2, n2 = self.group_control.get_mean_std_count()
        s = np.sqrt(((n1-1)*(s1**2)+(n2-1)*(s2**2))/(n1+n2-2))
        d = (m1 - m2) / s
        self.effect_size = d
        self.variance = (n1+n2)/(n1*n2) + d**2/(2*(n1+n2))
        self.standard_error = math.sqrt(self.variance)

    def hedge_g(self):
        self.cohen_d()
        n1, n2 = self.group_experimental.count, self.group_control.count
        j = (1-3/(4*(n1+n2)-9))
        g_star = j * self.effect_size
        self.effect_size = g_star
        self.variance = j**2 * self.variance
        self.standard_error = math.sqrt(self.variance)
    
    def risk_ratio(self):
        raise NotImplementedError()

    def get_effect_size(self):
        return self.effect_size

    def get_variance(self):
        return self.variance
    
    def get_standard_error(self):
        return self.standard_error

    def get_confidence_intervals(self):
        # 95% confidence intervals
        if self.data_type == self.num:
            lower_limits = self.effect_size - 1.96 * self.standard_error
            upper_limits = self.effect_size + 1.96 * self.standard_error
        elif self.data_type == self.cate:
            ln_lower_limits = np.log(self.effect_size) - 1.96 * self.standard_error
            ln_upper_limits = np.log(self.effect_size) + 1.96 * self.standard_error
            lower_limits = np.exp(ln_lower_limits)
            upper_limits = np.exp(ln_upper_limits)
        return lower_limits, upper_limits

class Center(object):
    """ meta analysis study, used to caculate effect size and variance.

    Attributes:
        name: str, center name
        shape: group's mean shape, used to check center consistency
        group_dict: dict of Group instance, {name1: group1, ...}

    Function:
        check_groups(): check groups consistency
        build_dict(groups): build dict for better index
        check_label(label): check whether has group of specific label in this center
        gen_study(label1, label2, method, index): generate Study instance
        gen_region_study(label1, label2, method, mask, region_label): generate Region Study
    """
    def __init__(self, name, groups):
        self.name = name
        self.check_groups(groups)
        self.build_dict(groups)

    def check_groups(self, groups):
        #check shape
        base_group = groups[0]
        class_name = type(base_group).__name__
        for group in groups:
            #check Group class
            assert type(group).__name__ == class_name
            
    def build_dict(self, groups):
        group_dict = {}
        for group in groups:
            label = group.get_label()
            if label not in group_dict:
                group_dict[label] = group
            else:
                raise ValueError("Two Groups have same label, please check or merge.")
        self.group_dict = group_dict

    def check_label(self, label):
        if label not in self.group_dict:
            print('Couln\'t found [label:{}] group in [center:{}]'.format(label, self.name))
            return False
        return True

    def gen_study(self, label1, label2, method):
        if not self.check_label(label1):
            return
        elif not self.check_label(label2):
            return
        else:
            group_experimental = self.group_dict[label1]
            group_control = self.group_dict[label2]
            return Study(self.name, method,
                         group_experimental, group_control)

class Centers(object):
    ''' Simply hold list of Center instance.
    Attributes:
        center_list: list of Center instances.
    Function:
        gen_studies(label1, label2, method): return list of Study instances.
    '''
    def __init__(self, center_list):
        super().__init__()
        self.center_list = center_list

    def gen_studies(self, label1, label2, method):
        studies = []
        for center in self.center_list:
            study = center.gen_study(label1, label2, method)
            studies.append(study)
        return studies
