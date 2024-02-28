""""
Script to load basis images and save to .mat files (as 40 and 70 keV VMI), for image
registration in MATLAB
"""
import imageio.v2 as imageio
import scipy
import os
import numpy as np
from matplotlib import pyplot as plt
import re
from catsim import GetMu

directory_water = 'C:/Users/Karin/Documents/outputs/water/water_one/m_pt148'
directory_iodine = 'C:/Users/Karin/Documents/outputs/iodine/iodine_one/m_pt148'
E = 70

mu_water = GetMu('water', E)[0]
mu_air = GetMu('air', E)[0]
mu_pe = GetMu('polyethylene', E)[0]
mu_pvc = GetMu('pvc', E)[0]

waio_to_pepvc = np.linalg.inv(np.array([[950.48, 1180.8], [-19.49, 303.14]])) #for correction of wrong change of basis

def HU(mu):
    hounsfield = 1000 * (mu - mu_water)/(mu_water-mu_air)
    return hounsfield

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

filenames = os.listdir(directory_water)
filenames.sort(key=natural_keys)

dicoms = []
for filename in filenames:
    f_w = os.path.join(directory_water, filename)
    a_water = imageio.imread(f_w)
    f_i = os.path.join(directory_iodine, filename)
    a_iodine = imageio.imread(f_i)
    a_pe = waio_to_pepvc[0][0] * a_water + waio_to_pepvc[0][1] * a_iodine
    a_pvc = waio_to_pepvc[1][0] * a_water + waio_to_pepvc[1][1] * a_iodine
    vmi = HU(a_pe * mu_pe + a_pvc * mu_pvc)

    dicoms.append(vmi)

dicom_array = np.stack(dicoms,0)

plt.imshow(dicom_array[80,:,:], cmap='gray', clim=[-1000, 1000])
plt.show()

mat_dict = {'vmi_{}kev'.format(E): dicom_array}
scipy.io.savemat('dicoms_pepvc_corr/dicoms_{}kev/{}kev_VMIs_pt148.mat'.format(E, E), mat_dict)