function generate_vp_file(phtName, materialList, sz, ps)
% Aim: to generate .vp file for voxelized phantom
% sz: phantom size (dimension), ps: voxelsize
% Mingye Wu, GE Research

nMaterial = numel(materialList);
fid = fopen([phtName, '.vp'],'wt');
fprintf(fid, sprintf('vp.n_materials = %d;\n\n', nMaterial));
for ii=1:nMaterial
    fprintf(fid, sprintf('vp.mat_name{%d} = ''%s'';\n', ii, materialList{ii}));
    fprintf(fid, sprintf('vp.volumefractionmap_filename{%d} = ''%s.density_%d'';\n', ii, phtName, ii));
    fprintf(fid, sprintf('vp.volumefractionmap_datatype{%d} = ''float'';\n', ii));
    fprintf(fid, sprintf('vp.cols{%d} = %d;\n', ii, sz(1)));
    fprintf(fid, sprintf('vp.rows{%d} = %d;\n', ii, sz(2)));
    fprintf(fid, sprintf('vp.slices{%d} = %d;\n', ii, sz(3)));
    fprintf(fid, sprintf('vp.x_size{%d} = %g;\n', ii, ps(1)));
    fprintf(fid, sprintf('vp.y_size{%d} = %g;\n', ii, ps(2)));
    fprintf(fid, sprintf('vp.z_size{%d} = %g;\n', ii, ps(3)));
    fprintf(fid, sprintf('vp.x_offset{%d} = %g;\n', ii, sz(1)/2+0.5));
    fprintf(fid, sprintf('vp.y_offset{%d} = %g;\n', ii, sz(2)/2+0.5));
    fprintf(fid, sprintf('vp.z_offset{%d} = %g;\n', ii, sz(3)/2+0.5));
    fprintf(fid, '\n');
end
fclose(fid);

end

