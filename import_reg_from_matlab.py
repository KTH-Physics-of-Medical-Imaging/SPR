"""
Script to import registered images from Matlab (40 keV images and 70 keV images), and stack VMI's
and SPR maps from different phantoms.
"""
import numpy as np
from matplotlib import pyplot as plt
import mat73
import re
import os

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

dicom_path_70 = 'C:/Users/Karin/Documents/MATLAB/XCAT/ICRU_pepvc_corr_dataset/70kev'
dicom_path_40 = 'C:/Users/Karin/Documents/MATLAB/XCAT/ICRU_pepvc_corr_dataset/40kev'

filenames_70 = os.listdir(dicom_path_70)
filenames_70.sort(key=natural_keys)

filenames_40 = os.listdir(dicom_path_40)
filenames_40.sort(key=natural_keys)

dicoms = []
for filename in filenames_70:
    f = os.path.join(dicom_path_70, filename)
    vmi_dict = mat73.loadmat(f)
    dicom = vmi_dict['I']
    dicom = dicom.astype('float32')
    dicoms.append(dicom)

VMIs_70 = np.concatenate(dicoms, 0, dtype='float32')
np.save('ICRU_pepvc_corr_VMIs_70', VMIs_70)
del VMIs_70

dicoms = []
for filename in filenames_40:
    f = os.path.join(dicom_path_40, filename)
    vmi_dict = mat73.loadmat(f)
    dicom = vmi_dict['I']
    dicom = dicom.astype('float32')
    dicoms.append(dicom)

VMIs_40 = np.concatenate(dicoms, 0, dtype='float32')

np.save('ICRU_pepvc_corr_VMIs_40', VMIs_40)

reg_SPR_path = 'C:/Users/Karin/Documents/MATLAB/XCAT/ICRU_pepvc_corr_dataset/SPR'

filenames = os.listdir(reg_SPR_path)
filenames.sort(key=natural_keys)

SPR_arrays = []
for filename in filenames:
    f = os.path.join(reg_SPR_path, filename)
    SPR_dict = mat73.loadmat(f)
    reg_SPR_array = SPR_dict['SPR_map']
    SPR_arrays.append(reg_SPR_array)

SPR_maps = np.concatenate(SPR_arrays, 0, dtype='float32')
np.save('ICRU_pepvc_corr_SPR_maps', SPR_maps)
del SPR_maps, SPR_arrays
