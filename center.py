import ast
import csv
import os
import re
import xml.etree.ElementTree as ET

import nibabel as nib
import nilearn as nil
import numpy as np
import pandas as pd
from nibabel import nifti1
from nibabel.freesurfer.io import read_morph_data
from nilearn import image
from pandas.errors import EmptyDataError

import utils

def load_array(path, dtype=np.float32):
    nii = nib.load(path)
    array = np.asarray(nii.dataobj)
    array = array.astype(dtype)
    array = np.nan_to_num(array)
    return array

class Person(object):
    def __init__(self, filename, label):
        super(Person, self).__init__()
        self.filename = filename
        self.label = label

class Center(object):
    """docstring for Center

    Attributes:
        file_dir: string, center's dir
        filenames: string, filepath to a csv file contains all person's no and their label
        use_nii: bool, whether to load nii
        use_csv: bool, whether to load csv
        use_xml: bool, whether to load xml
        persons: list of Person
    """

    def __init__(self, file_dir, filenames='origin.csv'):
        name = file_dir[file_dir.rfind('/')+1:]
        self.file_dir = file_dir
        self.name = name
        self.filenames = filenames
        self.persons = self.load_persons()

    def load_persons(self):
        """get list of Person

        Returns:
            list of Person
        """
        persons = []
        csv_path = os.path.join(self.file_dir, self.filenames)
        #get person's filename in txt file
        df = pd.read_csv(csv_path, index_col=0)
        for index, value in df.iterrows():
            filename = index
            label = value['label']
            _person = Person(filename, label)
            persons.append(_person)
        return persons

    def save_labels(self, filename):
        path = os.path.join(self.file_dir, filename)
        with open(path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['filename', 'label'])
            writer.writeheader()
            for person in self.persons:
                writer.writerow({'filename':person.filename,
                                 'label':person.label})

    def get_by_label(self, label, persons=None):
        """get list of Person in specified label

        Returns:
            list of Person
        """
        filtered = []
        if persons is None:
            if self.persons:
                for person in self.persons:
                    if person.label == label:
                        filtered.append(person)
        else:
            for person in persons:
                if person.label == label:
                    filtered.append(person)
        return filtered

    def get_by_gender_and_label(self, gender, label):
        genders, _ = self.get_genders()
        persons = self.persons
        filtered = []
        for g, person in zip(genders, persons):
            if g == gender and person.label == label:
                filtered.append(person)
        return filtered

    def create_stat_nii(self, label, temp_nii,
                        nii_dir='mri_smoothed_removed'):
        out_dir = os.path.join(self.file_dir, nii_dir, '{}'.format(label))
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        group1_data, _ = self.get_nii_pathes(label=label, nii_prefix=nii_dir+'/{}.nii')
        datas = utils.load_arrays(group1_data, dtype=np.float16)
        mean, std, _ = utils.cal_mean_std_n(datas)
        mean_path = os.path.join(out_dir, 'mean')
        std_path = os.path.join(out_dir, 'std')
        utils.gen_nii(mean, temp_nii, mean_path)
        utils.gen_nii(std, temp_nii, std_path)
    
    def create_stat_gii(self, label, gii_dir='surf', prefix='surf/s15.mesh.thickness.resampled_32k.removed.{}.gii'):
        out_dir = os.path.join(self.file_dir, gii_dir)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        out_dir = os.path.join(out_dir, '{}'.format(label))
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        pathes, _ = self.get_resampled_gii_pathes(label=label, prefix=prefix)
        datas = []
        if pathes is not None:
            for path in pathes:
                # For processed data
                #tmp = nil.surface.load_surf_data(path)
                #datas.append(tmp[-1])
                # original cat12 output
                datas.append(nib.load(path).get_arrays_from_intent(0)[0].data)
            mean, std, _ = utils.cal_mean_std_n(datas)
            mean_path = os.path.join(out_dir, 'mean')
            std_path = os.path.join(out_dir, 'std')
            np.save(mean_path, mean)
            np.save(std_path, std)

    def create_dir(self, dir_name):
        os.mkdir(os.path.join(self.file_dir, dir_name))

    def create_rgmv_csv(self, mask, label=None,
                        nii_prefix='mri_smoothed/{}.nii',
                        gmv_csv_prefix='roi_gmv/{}.csv'):
        if label is None:
            persons = self.persons
        else:
            persons = self.get_by_label(label)
        if len(persons) == 0:
            return None, None
        for person in persons:
            nii_path = os.path.join(self.file_dir,
                                    nii_prefix.format(person.filename))
            if not os.path.exists(nii_path):
                print('No nii file:{}:{}'.format(self.name, person.filename))
                continue
            csv_path = os.path.join(self.file_dir,
                                    gmv_csv_prefix.format(person.filename))
            if not os.path.exists(csv_path):
                nii = nib.load(nii_path)
                volumes = mask.get_all_masked_volume(nii)

                with open(csv_path, 'w', newline='') as file:
                    fieldnames = ['ID', 'GMV']
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    for k, v in volumes.items():
                        writer.writerow({'ID': k, 'GMV': v})

    def create_rct_csv(self, label=None,
                        cortical_id_path='./data/mask/cortical_id.csv',
                        cat_roi_prefix='label/catROIs_{}.xml',
                        ct_csv_prefix='roi_ct/{}.csv'):
        id_df = pd.read_csv(cortical_id_path, index_col=0)
        
        if label is None:
            persons = self.persons
        else:
            persons = self.get_by_label(label)
        if len(persons) == 0:
            return None, None
        for person in persons:
            xml_path = os.path.join(self.file_dir,
                                    cat_roi_prefix.format(person.filename))
            if not os.path.exists(xml_path):
                print('No catROIs file:{}:{}'.format(self.name,person.filename))
                continue
            csv_path = os.path.join(self.file_dir,
                                    ct_csv_prefix.format(person.filename))
            report = ET.parse(xml_path)
            root = report.getroot()
            names = root.findall('./aparc_BN_Atlas/names')

            thickness = root.find('./aparc_BN_Atlas/data/thickness')
            thickness = thickness.text.replace(' ', ',')
            thickness = thickness.replace('NaN', '-1')
            
            thickness_list = ast.literal_eval(thickness)

            with open(csv_path, 'w', newline='') as file:
                fieldnames = ['ID', 'CT']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for (item, thickness) in zip(names[0].findall('item'), thickness_list):
                    name = item.text
                    if name[0].lower() == name[-1].lower():
                        if '/' in name:
                            name = name.replace('/', '-')
                        writer.writerow({'ID': id_df.loc[name]['ID'], 'CT': thickness})

    def create_rct_csv_by_gii(self, gii_mask,
                              label=None, csv_dirname='aal',
                              surf_prefix='surf/s15.mesh.thickness.resampled_32k.{}.gii'):
        if label is None:
            persons = self.persons
        else:
            persons = self.get_by_label(label)
        if len(persons) == 0:
            return None, None
        for person in persons:
            dir_path = os.path.join(self.file_dir, csv_dirname)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            csv_path = os.path.join(dir_path, '{}.csv'.format(person.filename))
            gii_path = os.path.join(self.file_dir,
                                    surf_prefix.format(person.filename))

            if not os.path.exists(csv_path):
                gii = nib.load(gii_path)
                means = gii_mask.get_all_masked_mean(gii)

                with open(csv_path, 'w', newline='') as file:
                    fieldnames = ['ID', 'CT']
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    for k, v in means.items():
                        writer.writerow({'ID': k, 'CT': v})

    def get_nii_pathes(self, label=None, nii_prefix='mri/wm{}.nii'):
        pathes = []
        labels = []
        if label is None:
            persons = self.persons
        else:
            persons = self.get_by_label(label)
        if len(persons) == 0:
            return None, None
        for person in persons:
            pathes.append(os.path.join(self.file_dir,
                          nii_prefix.format(person.filename)))
            labels.append(person.label)
        labels = np.asarray(labels)
        return pathes, labels

    def create_smoothed_image(self, fwhm=4, in_prefix='mri/mwp1{}.nii',
                              out_prefix='mri_smoothed/{}.nii'):
        for person in self.persons:
            path = os.path.join(self.file_dir,
                                in_prefix.format(person.filename))
            origin_nii = nib.load(path)
            nii = image.smooth_img(origin_nii, fwhm)
            self.save_nii(person, nii, nii_prefix=out_prefix)

    def save_nii(self, person, nii, nii_prefix='mri_smoothed_removed/{}.nii'):
        path = os.path.join(self.file_dir,
                            nii_prefix.format(person.filename))
        nifti1.save(nii, path)

    def get_average_thickness_by_person(self, person, cat_prefix='report/cat_{}.xml'):
        xml_path = os.path.join(self.file_dir,
                                    cat_prefix.format(person.filename))
        report = ET.parse(xml_path)
        root = report.getroot()
        cgw = root.findall('./subjectmeasures/dist_thickness/item')
        act = [float(i) for i in cgw[0].text.replace('[', '').replace(']', '').split()][0]
        return act

    def get_average_thickness(self, label=None, cat_prefix='report/cat_{}.xml'):
        features = []
        labels = []
        if label is None:
            persons = self.persons
        else:
            persons = self.get_by_label(label)
        if len(persons) == 0:
            return None, None
        for person in persons:
            value = self.get_average_thickness_by_person(person, cat_prefix)
            features.append(value)
            labels.append(person.label)
        features = np.asarray(features)
        labels = np.asarray(labels)
        return features, labels

    def get_tiv_by_person(self, person, cat_prefix='report/cat_{}.xml'):
        xml_path = os.path.join(self.file_dir,
                                    cat_prefix.format(person.filename))
        report = ET.parse(xml_path)
        root = report.getroot()
        vol_tiv = float(root.findall('./subjectmeasures/vol_TIV')[0].text)
        return vol_tiv
    
    def get_cgw_by_person(self, person, cat_prefix='report/cat_{}.xml'):
        xml_path = os.path.join(self.file_dir,
                                    cat_prefix.format(person.filename))
        report = ET.parse(xml_path)
        root = report.getroot()
        cgw = root.findall('./subjectmeasures/vol_abs_CGW')
        tmp = [float(i) for i in cgw[0].text.replace('[', '').replace(']', '').split()]
        return tmp

    def get_tivs_cgws(self, label=None, cat_prefix='report/cat_{}.xml'):
        features = []
        labels = []
        if label is None:
            persons = self.persons
        else:
            persons = self.get_by_label(label)
        if len(persons) == 0:
            return None, None
        for person in persons:
            vol_tiv = self.get_tiv_by_person(person, cat_prefix)
            cgw = self.get_cgw_by_person(person, cat_prefix)
            cgw.insert(0, vol_tiv)
            
            features.append(cgw[0:4])
            labels.append(person.label)
        features = np.asarray(features)
        labels = np.asarray(labels)
        return features, labels
    
    def get_tivs(self, label=None, cat_prefix='report/cat_{}.xml'):
        features, labels = self.get_tivs_cgws(label, cat_prefix)
        if labels is not None:
            tivs = features[:,0]
        else:
            tivs = None
        return tivs, labels

    def get_csfs(self, label=None, cat_prefix='report/cat_{}.xml'):
        features, labels = self.get_tivs_cgws(label, cat_prefix)
        if labels is not None:
            csfs = features[:,1]
        else:
            csfs = None
        return csfs, labels

    def get_gmvs(self, label=None, cat_prefix='report/cat_{}.xml'):
        features, labels = self.get_tivs_cgws(label, cat_prefix)
        if labels is not None:
            gmvs = features[:,2]
        else:
            gmvs = None
        return gmvs, labels

    def get_wmvs(self, label=None, cat_prefix='report/cat_{}.xml'):
        features, labels = self.get_tivs_cgws(label, cat_prefix)
        if labels is not None:
            wmvs = features[:,3]
        else:
            wmvs = None
        return wmvs, labels

    def get_image_quality(self, label=None, cat_prefix='report/cat_{}.xml',
                          threshold=6, verbose=False):
        features = []
        labels = []
        if label is None:
            persons = self.persons
        else:
            persons = self.get_by_label(label)
        if len(persons) == 0:
            return None, None
        for person in persons:
            xml_path = os.path.join(self.file_dir,
                                    cat_prefix.format(person.filename))
            report = ET.parse(xml_path)
            root = report.getroot()

            resolution = 10.5 - float(root.findall('./qualityratings/res_RMS')[0].text)
            noise = 10.5 - float(root.findall('./qualityratings/NCR')[0].text)
            bias = 10.5 - float(root.findall('./qualityratings/ICR')[0].text)
            iqr = 10.5 - float(root.findall('./qualityratings/IQR')[0].text)
            qualties = [resolution, noise, bias, iqr]

            if verbose:
                if resolution < threshold:
                    print('{}:Res scores:{}'.format(person.filename, resolution))
                if noise < threshold:
                    print('{}:Noise scores:{}'.format(person.filename, noise))
                if bias < threshold:
                    print('{}:Bias scores:{}'.format(person.filename, bias))
                if iqr < threshold:
                    print('{}:IQR scores:{}'.format(person.filename, iqr))
            
            features.append(qualties)
            labels.append(person.label)
        features = np.asarray(features)
        labels = np.asarray(labels)
        return features, labels

    def get_resolution(self, label=None, cat_prefix='report/cat_{}.xml'):
        features, labels = self.get_image_quality(label, cat_prefix)
        if labels is not None:
            resolution = features[:,0]
        else:
            resolution = None
        return resolution, labels
    
    def get_noise(self, label=None, cat_prefix='report/cat_{}.xml'):
        features, labels = self.get_image_quality(label, cat_prefix)
        if labels is not None:
            noise = features[:,1]
        else:
            noise = None
        return noise, labels
    
    def get_bias(self, label=None, cat_prefix='report/cat_{}.xml'):
        features, labels = self.get_image_quality(label, cat_prefix)
        if labels is not None:
            bias = features[:,2]
        else:
            bias = None
        return bias, labels
    
    def get_iqr(self, label=None, cat_prefix='report/cat_{}.xml'):
        features, labels = self.get_image_quality(label, cat_prefix)
        if labels is not None:
            iqr = features[:,3]
        else:
            iqr = None
        return iqr, labels

    def get_cortical_thickness(self, label=None,
                               surf_prefix='surf/s15.mesh.thickness.resampled_32k.{}.gii'):
        features = []
        labels = []
        if label is None:
            persons = self.persons
        else:
            persons = self.get_by_label(label)
        if len(persons) == 0:
            return None, None
        for person in persons:
            ct_path = os.path.join(self.file_dir,
                                   surf_prefix.format(person.filename))
            ct_gii = nib.load(ct_path)
            ct_darray = ct_gii.get_arrays_from_intent(0)[0]
            features.append(ct_darray.data)
            labels.append(person.label)
        features = np.asarray(features)
        labels = np.asarray(labels)
        return features, labels
    
    def get_cortical_thickness_pathes(self, label=None,
                                      surf_prefix='surf/s15.mesh.thickness.resampled_32k.{}.gii'):
        ct_pathes = []
        labels = []
        if label is None:
            persons = self.persons
        else:
            persons = self.get_by_label(label)
        if len(persons) == 0:
            return None, None
        for person in persons:
            ct_path = os.path.join(self.file_dir,
                                   surf_prefix.format(person.filename))
            ct_pathes.append(ct_path)
            labels.append(person.label)
        ct_pathes = np.asarray(ct_pathes)
        labels = np.asarray(labels)
        return ct_pathes, labels

    def get_presonal_info_values_by_person(self, person,
                                            personal_info_prefix='personal_info/{}.csv'):
        csv_path = os.path.join(self.file_dir,
                                personal_info_prefix.format(person.filename))
        df = pd.read_csv(csv_path)
        values = df.to_numpy().flatten()
        if len(values) == 3:
            values = np.append(values, np.nan)
        if np.isnan(values[-1]):
            print(self.file_dir, person.filename)
        return values

    def get_presonal_info_values(self, label=None,
                                personal_info_prefix='personal_info/{}.csv'):
        # get person's age, male, female, MMSE
        features = []
        labels = []
        if label is None:
            persons = self.persons
        else:
            persons = self.get_by_label(label)
        if len(persons) == 0:
            return None, None
        for person in persons:
            values = self.get_presonal_info_values_by_person(person, personal_info_prefix)
            features.append(values)
            labels.append(person.label)
        features = np.asarray(features)
        labels = np.asarray(labels)
        return features, labels
    
    def get_ages(self, label=None,
                 personal_info_prefix='personal_info/{}.csv'):
        features, labels = self.get_presonal_info_values(label, personal_info_prefix)
        if labels is not None:
            ages = features[:,0]
        else:
            ages = None
        return ages, labels

    def get_males(self, label=None,
                    personal_info_prefix='personal_info/{}.csv'):
        features, labels = self.get_presonal_info_values(label, personal_info_prefix)
        if labels is not None:
            genders = features[:,1]
        else:
            genders = None
        return genders, labels
    
    def get_females(self, label=None,
                    personal_info_prefix='personal_info/{}.csv'):
        features, labels = self.get_presonal_info_values(label, personal_info_prefix)
        if labels is not None:
            genders = features[:,2]
        else:
            genders = None
        return genders, labels

    def get_filenames(self, label=None):
        filenames = []
        labels = []
        if label is None:
            persons = self.persons
        else:
            persons = self.get_by_label(label)
        if len(persons) == 0:
            return None, None
        for person in persons:
            filenames.append(person.filename)
        return filenames, labels

    def get_MMSEs(self, label=None,
                 personal_info_prefix='personal_info/{}.csv'):
        features, labels = self.get_presonal_info_values(label, personal_info_prefix)
        if labels is not None:
            MMSEs = features[:,-1]
        else:
            MMSEs = None
        return MMSEs, labels

    def get_csv_values(self, persons=None, prefix='roi_gmv/{}.csv', flatten=False):
        features = []
        labels = []

        if persons is None:
            persons = self.persons
        if len(persons) == 0:
            return None, None, None
        
        for person in persons:
            csv_path = os.path.join(self.file_dir,
                                    prefix.format(person.filename))
            try:
                df = pd.read_csv(csv_path, index_col=0)
                if flatten:
                    features.append(df.to_numpy().flatten())
                else:
                    features.append(df.to_numpy())
                labels.append(person.label)
                ids = df.index.tolist()
            except EmptyDataError:
                print(person.filename)
        features = np.stack(features)
        labels = np.stack(labels)
        return features, labels, ids
    
    def get_csv_df(self, label=None, prefix='roi_gmv/{}.csv'):
        dfs = []
        labels = []
        if label is None:
            persons = self.persons
        else:
            persons = self.get_by_label(label)
        if len(persons) == 0:
            return None, None
        for person in persons:
            csv_path = os.path.join(self.file_dir,
                                    prefix.format(person.filename))
            df = pd.read_csv(csv_path, index_col=0)
            dfs.append(df)
            labels.append(person.label)
        labels = np.stack(labels)
        return dfs, labels

    def get_resampled_gii_pathes(self, label, prefix='surf/s15.mesh.thickness.resampled_32k.removed.{}.gii'):
        pathes = []
        labels = []
        if label is None:
            persons = self.persons
        else:
            persons = self.get_by_label(label)
        if len(persons) == 0:
            return None, None
        for person in persons:
            pathes.append(os.path.join(self.file_dir,
                          prefix.format(person.filename)))
            labels.append(person.label)
        labels = np.asarray(labels)
        return pathes, labels

    def load_msn_nii_array(self, label, _dir='mri_smoothed'):
        path = os.path.join(self.file_dir, _dir, '{}'.format(label))
        mean_path = os.path.join(path, 'mean.nii')
        std_path = os.path.join(path, 'std.nii')
        mean = load_array(mean_path)
        std = load_array(std_path)
        count = len(self.get_by_label(label))
        return mean, std, count

    def load_msn_array(self, label, _dir='surf'):
        path = os.path.join(self.file_dir, _dir, '{}'.format(label))
        mean_path = os.path.join(path, 'mean.npy')
        std_path = os.path.join(path, 'std.npy')
        mean = np.load(mean_path)
        std = np.load(std_path)
        count = len(self.get_by_label(label))
        return mean, std, count
