%Script to merge two XCAT phantom files into one and
%reduce density maps to 2 through a material decomposition in PE and PVC
clear
clc
fclose('all');

%%
%Paths to the two phantom files, hardcoded for phantom file
%names in format phantomname_startslice_endslice, folder for voxelized
%phantom output and folder for the raw meged phantoms (needed for image
%registration)

pht_path = 'C:/Users/Karin/XCAT/Setup1/';

pht_name = 'vmale50';
slices = ["3899-4032" "4033-4205"];

new_pht_name = 'vmale50';
new_pht_folder = 'C:/Users/Karin/XCAT/Trainingdata_test/';

foldername = [new_pht_folder new_pht_name '_2MTS_vox'];
mkdir(foldername);

mergedphantoms_folder = 'C:/Users/Karin/XCAT/MergedPhantoms_bgAdipose';
mkdir(mergedphantoms_folder)

%%
%Read dimensions and properties from log file, load phantom files
for i=1:2
    log_name = [pht_name "_" slices(i) "_log"];
    log_name = join(log_name, "");
    log_name = fullfile(pht_path, log_name);

    fid = fopen(log_name);
    log_line = fgetl(fid);

    while ischar(log_line)
        if contains(log_line, 'pixel width')
            log_line = split(log_line);
            if size(log_line,1)==6
                pixel_width = log_line{5};
            end
        end

        if contains(log_line, 'slice width')
            log_line = split(log_line);
            if size(log_line,1)==6
                slice_width = log_line{5};
            end
        end

        if contains(log_line, 'array_size')
            log_line = split(log_line);
            xy_dims = log_line{4};   
        end

        if contains(log_line, 'starting slice number')
            log_line = split(log_line);
            start_slice = str2double(log_line{6});
            log_line = fgetl(fid);
            log_line = split(log_line);
            end_slice = str2double(log_line{6});
            nr_slices{i} = end_slice - start_slice + 1;
        end
   
        log_line = fgetl(fid);
    end
    act_name = [pht_name "_" slices(i) "_act_1.bin"];
    act_name = join(act_name, "");
    fname = fullfile(pht_path, act_name);


    fid = fopen(fname);
    act_file{i} = fread(fid, 'float32');
    act_file{i} = reshape(act_file{i}, [str2double(xy_dims), str2double(xy_dims), nr_slices{i}]);
    
end

%%
%Merge into one phantom file
merged = cat(3, act_file{1}, act_file{2});
fname = fullfile(mergedphantoms_folder, pht_name);
fid = fopen(fname, 'wb');
fwrite(fid, merged, 'single');
fclose(fid);
merged = permute(merged, [2 1 3]);

%%
%Make lists for phantom dimensions and pixel sizes

pixel_width = str2double(pixel_width);
slice_width = str2double(slice_width);
xy_dims = str2double(xy_dims);
nr_slices = sum([nr_slices{:}]);

sz = [xy_dims xy_dims nr_slices];
ps = [pixel_width pixel_width slice_width];

%%
% Prepare material decomposition PE and PVC. Dictionaries associate
% material indices with linear attenuation coefficients at 40 and 70 keV
% 0 air, 1 water, 2 muscle, 3 lung, 4 dry spine, 6 adipose, 7 blood, 14 skull, 15 cartilage, 16 brain, 
% 27 skin, 29 eye lens, 31 red marrow, 36 grey matter, 37 white matter

mt_ids = [0 1 2 3 4 6 7 14 15 16 27 29 31 36 37];

mu_40 = [0 0.26843816 0.28211325 0.07022181 0.7059226 0.22832586 0.28793606 0.92604864 0.3116322 0.2811304 0.284054 ...
    0.27727273 0.26302722 0.28333348 0.27834764];

mu_70 = [0 0.19260076 0.20088093 0.04984349 0.3252056 0.17800109 0.20340341 0.3895827 0.21253248 0.19990811 0.20660731 ...
    0.20199855 0.19479173 0.20034447, 0.19924009];

mt2mu_40kev = dictionary(mt_ids, mu_40);
mt2mu_70kev = dictionary(mt_ids, mu_70);

mu_pe_pvc = [0.2115951 1.027087; 0.175358 0.37639758]; %A vector, PE, PVC

%%
%Solve equation system Ax = B and obtain density maps for PE and PVC
a_pe = zeros([xy_dims, xy_dims, nr_slices]);
a_pvc = zeros([xy_dims, xy_dims, nr_slices]);

for i = 1:length(mt_ids)
    mt_idx = find(merged == mt_ids(i));
    mt_mu_pixel = [mt2mu_40kev(mt_ids(i)); mt2mu_70kev(mt_ids(i))];
    a_vec = linsolve(mu_pe_pvc, mt_mu_pixel);
    a_pe(mt_idx) = a_vec(1);
    a_pvc(mt_idx) = a_vec(2);
end

dens_maps={a_pe a_pvc};
catsim_mts = ["polyethylene" "pvc"];

%%
%Write density files
for i=1:2
    fname = sprintf('%s.density_%d',[foldername '/' new_pht_name], i);
    fid = fopen(fname, 'wb');
    fwrite(fid, dens_maps{i}, 'single');
    fclose(fid);
end

%%
%Generate the vp file
generate_vp_file([foldername '/' new_pht_name], catsim_mts, sz, ps);

