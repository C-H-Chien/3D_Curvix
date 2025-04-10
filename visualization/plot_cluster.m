clear;
close all;

% === Settings ===
validation_view_index = 42;
Dataset_Path = "/gpfs/data/bkimia/Datasets/ABC-NEF/";
Object_Name = "00000006/";
Image_Path = fullfile(Dataset_Path, Object_Name, "train_img/");
img_filename = fullfile(Image_Path, sprintf('%d_colors.png', validation_view_index));
img = imread(img_filename);

% === Cluster coordinates ===
cluster0 = [ ...
   332.173, 468.692;
   332.329, 468.834;
   332.477, 468.897;
   332.792, 469.122;
   332.937, 469.185;
   333.093, 469.328;
   333.244, 469.395;
   333.554, 469.611;
   333.698, 469.669;
   333.852, 469.808];


% cluster1 = [ ...
%     419.813, 364.546;
%     420.061, 364.531;
%     420.311, 364.517];
% 
% cluster2 = [ ...
%     365.122, 312.79;
%     365.407, 312.711];


% === Plot ===
figure; imshow(img); hold on;

% Plot each cluster with different color
scatter(cluster0(:,1), cluster0(:,2), 60, 'r', 'filled', 'DisplayName', 'Cluster 0');
% scatter(cluster1(:,1), cluster1(:,2), 60, 'g', 'filled', 'DisplayName', 'Cluster 1');
% scatter(cluster2(:,1), cluster2(:,2), 60, 'b', 'filled', 'DisplayName', 'Cluster 2');

legend('show');
title(sprintf('Clusters on Image %d', validation_view_index));
