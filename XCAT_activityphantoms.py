#Script to compute SPR maps from XCAT activity phantoms
import numpy as np
from matplotlib import pyplot as plt
import scipy
from CalcSPR import value_SPR
from catsim import rawread

#Dictionary material id - material name
mt_ids = list(range(1,38))
mts = ["ncat_water", "ncat_muscle", "ncat_lung", "ncat_dry_spine", "ncat_dry_rib", "ncat_adipose", "ncat_blood", "ncat_heart",
       "ncat_kidney", "ncat_liver", "ncat_lymph", "ncat_pancreas", "ncat_intestine", "ncat_skull", "ncat_cartilage", "ncat_brain",
       "ncat_spleen", "ncat_blood_with_1_1pct_iodine", "ncat_iron", "ncat_pmma", "ncat_aluminium", "ncat_titanium", "ncat_air",
       "ncat_graphite", "ncat_lead", "ncat_breast_mammary", "ncat_skin", "ncat_iodine", "ncat_eye_lens", "ncat_ovary", "ncat_red_marrow",
       "ncat_yellow_marrow", "ncat_testis", "ncat_thyroid", "ncat_bladder", "karin_grey_matter", "karin_white_matter"]

id_to_mt = {mt_ids[i]: mts[i] for i in range(len(mts))}

file_path = 'C:/Users/Karin/XCAT/MergedPhantoms_bgAdipose/vmale50_merged'
act_file = rawread(file_path, [], 'float')
nr_slices = 307
act_file = act_file.reshape(nr_slices, 1024, 1024)

phantom_materials = np.unique(act_file)
phantom_materials = np.delete(phantom_materials, 0)

for i in phantom_materials:
       SPR = value_SPR(id_to_mt[int(i)])
       print(SPR)
       act_file[act_file == i] = SPR


np.save('SPR_map_vmale50_ICRU', act_file)

matlab_dict = {'SPR_map': act_file}

scipy.io.savemat('SPR_map_vmale50_ICRU.mat', matlab_dict)

plt.imshow(act_file[36,:,:], cmap='gray')

plt.show()

