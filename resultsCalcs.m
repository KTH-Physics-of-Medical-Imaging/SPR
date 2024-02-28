%Aim: to make general and material-wise comparison of the predicted and true SPR.
%Plots and error calculations

clear 
clc

data = load('C:\Program och kod\SPRNetwork\Results\imgs\sr3_alt_2VMI_64_1_celebahq256new_for_check_vgg16_l1_9_350_2_1_1_2_train_ICRU_pepvc_corr_test_ICRU_pepvc_corr/6_data.mat');

predicted = getfield(data, 'prediction');
predicted = squeeze(predicted);

observed = getfield(data, 'observed');
observed = squeeze(observed);
observed = permute(observed, [2 3 1]);

truth = getfield(data, 'truth');
truth = squeeze(truth);

%%

figure, imshow(truth, [0, 1.5])
figure, imshow(predicted, [0, 1.5])


%%
diff = predicted - truth;
figure, imagesc(diff, [-0.25 0.25])
colormap('gray')
c=colorbar;
c.Label.String = '\Delta SPR';
c.FontSize = 12;
axis off
%%
imshow(observed(:,:,1), [0 70])
figure, imshow(observed(:,:,2), [0 70])

figure, imagesc(predicted, [0 1.5])
colormap('gray')
c=colorbar;
c.Label.String = 'SPR';
c.Label.FontSize = 12;
axis off

figure, imagesc(truth, [0 1.5])
colormap('gray')
c2=colorbar;
c2.Label.String = 'SPR';
c2.FontSize = 12;
axis off
%figure, imshow(predicted, [0 2])
%figure, imshow(truth, [0 2])

%%
%Line comparison and residuals
row = 100;
%res = abs(predicted(row,:)-truth(row,:));
res = predicted(row,:)-truth(row,:);

x = linspace(0,200,1024);
tiledlayout(2,1)

% Top plot
nexttile

plot(predicted(row,:), "--", 'DisplayName', 'Predicted SPR')
legend('-DynamicLegend', 'location', 'northwest', 'FontSize', 8)
hold all
plot(truth(row,:), 'DisplayName', 'True SPR')
xlabel('Pixel') 
ylabel('SPR')
xlim([0 256])
xticks(0:50:256)
ylim([-0.1 2])
yticks(0:1:2)

% Bottom plot
nexttile
plot(res)
xlabel('Pixel')
%ylabel('\mid \Delta SPR \mid')
ylabel(' \Delta SPR')
xlim([0 256])
xticks(0:50:256)
ylim([-0.15 0.15])


% Line indicator
figure, imshow(truth, [0 2])
sz = size(truth(:,:,1));
endx = min(sz(1), sz(2));
hold on
h(2) = plot([1 endx], [row row], 'r');
hold off

%%
figure, imshow(truth, [1.03814 1.03816])
%%
%Set indices for ROI-boxes in prediction and truth,
%x_[nr_ROI]_[nr_idx, 1 eller 2]
% truth: grey matter 1.0364, white matter 1.03867, ncat_brain = 1.03815,
% ncat_skull 1.4614.

corner_pts = [92, 95, 94, 98, 131, 134, 102, 106, 126, 129, 175, 180]; %white matter
%corner_pts = [125, 128, 128, 133, 99, 106, 184, 186, 135, 137, 182, 190]; %grey matter
%corner_pts = [135, 138, 144, 149, 100, 102, 132, 140, 158, 160, 145, 152]; %ncat_brain
%corner_pts = [166, 169, 122, 127, 110, 115, 76, 79, 71, 73, 123, 130]; %ncat_skull


x_1_1 = corner_pts(1);
x_1_2 = corner_pts(2);
y_1_1 = corner_pts(3);
y_1_2 = corner_pts(4);

x_2_1 = corner_pts(5);
x_2_2 = corner_pts(6);
y_2_1 = corner_pts(7);
y_2_2 = corner_pts(8);

x_3_1 = corner_pts(9);
x_3_2 = corner_pts(10);
y_3_1 = corner_pts(11);
y_3_2 = corner_pts(12);


ROI_pred = {predicted(x_1_1:x_1_2, y_1_1:y_1_2), predicted(x_2_1:x_2_2, y_2_1:y_2_2), predicted(x_3_1:x_3_2, y_3_1:y_3_2)};
ROI_truth = {truth(x_1_1:x_1_2, y_1_1:y_1_2), truth(x_2_1:x_2_2, y_2_1:y_2_2), truth(x_3_1:x_3_2, y_3_1:y_3_2)};


%%
%Check ROI's in image
check_im = truth;

check_im(x_1_1:x_1_2, y_1_1:y_1_2) = 10;
check_im(x_2_1:x_2_2, y_2_1:y_2_2) = 20;
check_im(x_3_1:x_3_2, y_3_1:y_3_2) = 30;

figure, imshow(check_im, [1.4 2])

%%
%Draw ROI's
figure, imshow(predicted, [0 2])
r1_white = drawrectangle('Position',[y_1_1, x_1_1, y_1_2-y_1_1, x_1_2-x_1_1],'Color','red');
r2_white = drawrectangle('Position', [y_2_1, x_2_1, y_2_2-y_2_1, x_2_2-x_2_1],'Color','blue');
r3_white = drawrectangle('Position', [y_3_1, x_3_1, y_3_2-y_3_1, x_3_2-x_3_1], 'Color', 'green');

%%
%RMSE
%All predicted values in ROI are included
RMSEs = cell(3,1);

for i=1:3
    RMSE = rmse(ROI_pred{i}, ROI_truth{i}, 'all');
    RMSEs{i} = RMSE*100;
    unique(ROI_truth{i})
end

%%
%Mean of RMSE
mean_RMSE = 0;
for i=1:3
    mean_RMSE = mean_RMSE + RMSEs{i};
end
mean_RMSE = mean_RMSE/3;

%%
%Relative error
%(Mean predicted SPR in the ROI as the predicted value)
REs = cell(3,1);

for i=1:3
    mean_pred = mean(ROI_pred{i}, 'all');
    ref_truth = unique(ROI_truth{i});
    RE = abs(ref_truth - mean_pred)/ref_truth;
    REs{i} = RE*100;
end 

%%
%Mean of relative error for ROI's
mean_RE = 0;

for i=1:3
    mean_RE = mean_RE + REs{i};
end
mean_RE = mean_RE/3;

%%
%Number of pixels in ROI
nr_pixels = cell(3,1);
total = 0;

for i=1:3
    nr_pixels{i} = numel(ROI_pred{i});
    total = total + nr_pixels{i};
end

%%
% Relative standard deviation (sample)
R_SD= cell(3,1);
mean_R_SD = 0;
for i=1:3
    mean_ROI = mean(ROI_pred{i}, 'all');
    delta = ROI_pred{i} - mean_ROI;
    delta = delta.^2;
    rel_st_dev = sqrt(sum(delta, 'all')/(numel(delta)-1));
    R_SD{i} = 100* rel_st_dev/unique(ROI_truth{i});
    mean_R_SD = mean_R_SD + R_SD{i};
end 
mean_R_SD = mean_R_SD/3;

%%
%Mean predicted SPR 
mean_SPR = cell(3,1);
for i=1:3
    mean_SPR{i} = mean(ROI_pred{i}, 'all');
    true_SPR = unique(unique(ROI_truth{i}));
end


