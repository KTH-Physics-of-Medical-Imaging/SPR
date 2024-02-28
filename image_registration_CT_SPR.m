%Script for image registration, SPR maps and CatSim CT dicoms - needs
%raw XCAT phantom for theoretical 70 keV VMI, dicom arrays
%(prepared with prepare_dicoms.py) and SPR maps (XCAT_activityphantoms.py)
clear 
clc
%%
%Read the SPR map files for the phantom, rotate to align with dicoms

SPR_path = 'C:/Users/Karin/SPRCalculations/Files';
myFiles = dir(fullfile(SPR_path,'*.mat'));
SPR_maps = cell(6,1);
remove = [1:13 295:307];

for k = 1:length(myFiles)
  baseFileName = myFiles(k).name;
  fullFileName = fullfile(SPR_path, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  SPR_map = load(fullFileName);
  SPR_map = getfield(SPR_map, "SPR_map");
  SPR_map = permute(SPR_map, [3,2,1]);
  SPR_map(:,:,remove) = [];
  SPR_map = flip(SPR_map, 3);
  SPR_map = imrotate(SPR_map, 180);
  SPR_maps{k} = SPR_map;
end

%%
%Select SPR map for registration
SPR_map = SPR_maps{6};

%%
%Read the merged raw activity phantom, to be transformed into theoretical 70 keV
%VMI
ph_path = 'C:/Users/Karin/XCAT/MergedPhantoms_bgAdipose/vmale50_merged';

nr_slices = 307;
xy_dims = 1024;

fid = fopen(ph_path);
ph_file = fread(fid, 'float32');
ph_file = reshape(ph_file, [xy_dims, xy_dims, nr_slices]);
ph_file = permute(ph_file, [1 2 3]);
ph_file = imrotate(ph_file, 180);

ph_file(:,:,remove) = [];
ph_file = flip(ph_file, 3);

%%
slice = 10;
imshow(ph_file(:,:,slice), [0 40])
figure, imshow(SPR_map(:,:,slice), [0 2])
%%
%Make dictionary from material id's in activity phantom to corresponding
%linear attenuation coefficient at 70 keV

mt_ids = [0 1 2 3 4 6 7 14 15 16 27 29 31 36 37];

mu_70 = [0 0.19260076 0.20088093 0.04984349 0.3252056 0.17800109 0.20340341 0.3895827 0.21253248 0.19990811 0.20660731 ...
    0.20199855 0.19479173 0.20034447, 0.19924009];

mt2mu_70kev = dictionary(mt_ids, mu_70);


%%
%Replace material id with corresponding HU
for i = 1:length(mt_ids)
    HU = 1000 * (mt2mu_70kev(mt_ids(i)) - mt2mu_70kev(1))/(mt2mu_70kev(1) - mt2mu_70kev(0));
    ph_file(ph_file == mt_ids(i)) = HU;
end

%%
%Load the VMI's

CT_path = 'C:\Users\Karin\SPRNetwork\dicoms_pepvc_corr\dicoms_70kev';
CT_filename = '70kev_VMIs_vm50.mat';

I = load(fullfile(CT_path, CT_filename));
I = getfield(I, "vmi70");
I = permute(I, [2, 3, 1]);

%Remove the FOV circle by setting the outside to HU for air
I(I<-2000) = -1000;

%%
n = 150;
figure, imshow(I(:,:,n), [-1000 1000])
figure, imshow(ph_file(:,:,n), [-1000 1000])

%%
%Compute a registration transform using the attenuation phantom
%and the CatSim VMI (adjusting the attenuation phantom and keeping dicom fixed)

[optimizer,metric] = imregconfig("multimodal");
optimizer.Epsilon = 1.5e-4;
optimizer.GrowthFactor = 1.01;
optimizer.MaximumIterations = 300;

tformSimilarity = imregtform(ph_file(:,:,170), I(:,:,170),"similarity",optimizer,metric);
Rfixed = imref2d(size(I));

%%
%Apply to the theoretical VMI for check
registeredSimilarity = imwarp(ph_file,tformSimilarity,OutputView=Rfixed);

%%
n = 50;
imshow(ph_file(:,:,n), [-1100 1100])
figure, imshow(registeredSimilarity(:,:,n), [-1100 1100])
figure, imshowpair(registeredSimilarity(:,:,n), I(:,:,n))

%%
%Apply the registration transform to SPR maps
for i=1:size(SPR_map, 3)
    SPR_map(:,:,i) = imwarp(SPR_map(:,:,i), tformSimilarity, Outputview=Rfixed);
end

%%
n= 200;
imagesc(SPR_map(:,:,n))
figure, imagesc(I(:,:,n))
figure, imshowpair(SPR_map(:,:,n), I(:,:,n))

%%
%Save registered SPR map
SPR_map = permute(SPR_map, [3, 1, 2]);
save('registeredSPR_vm50.mat','SPR_map');

%%
%Save dicoms with removed FOV rings
I = permute(I, [3, 1, 2]);
save('70kevdicoms_withoutFOVring_vm50.mat', 'I')
