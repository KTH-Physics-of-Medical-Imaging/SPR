# 

This repository contains data and code for 'Deep learning estimation of proton stopping power with photon-counting computed tomography', and contains scripts for converting XCAT phantoms to the format required by the PCCT simulation software, computation of ground truth SPR maps, neural network and results calculation. 

The network 'sr3_alt' is developed by Dennis Hein and builds upon a network presented in "Score-based generative modeling through stochastic differential equations" by Song et al. (2021).

'mergephantoms_2MD.m' takes two XCAT raw output files, merges them to one and decomposes the tissue materials into two density maps for PE and PVC. The CatSim scripts for generating PCCT basis images are proprietary and will not be shared. 'XCAT_activityphantoms.py' transforms the raw XCAT output into ground truth SPR maps, using 'CalcSPR'. 'prepare_dicoms' transforms the basis image output from CatSim into VMIs, which are then imported to Matlab for image registration with 'image_registration_CT_SPR.m'. 'import_reg_from_matlab.m' and 'prepare_dataset_nb.ipynb' are used to generate tensors from registered VMIs and SPR maps, which are then used as input to train the network with 'train_2VMI.py'. 'evaluate_2VMI.py' is used to estimate SPR maps from VMIs with the pre-trained network. All plots and errors are calculated with 'resultsCalcs.m', and the estimated SPR maps for which we reported errors can be found in 'estimated_SPR_maps.mat'. The Python scripts used for computing SPR and VMIs requires CatSim installed in the virtual environment, available at https://github.com/xcist.

