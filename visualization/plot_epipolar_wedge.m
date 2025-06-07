clear;
close all;

% Define dataset and image paths
Dataset_Path = "/gpfs/data/bkimia/Datasets/ABC-NEF/";
Object_Name = 8605;
Image_Path = fullfile(Dataset_Path, Object_Name, "train_img");


% Original edges
image_edges_our = [
30 524.536 366.277 0.159414;
30 525.023 366.354 0.146074;
30 525.512 366.423 0.137538;
];

image_edges_wacv = [

];


reprojection_our = [30 524.755 366.163 -0.985276];
reprojection_wacv = [30 531.947 381.47 0.579525];

epipoles_hypo1_our = [1648.99 2758.96    1];
angle_ranges_hypo1_our = [64.8199, 64.8479];
epipoles_hypo2_our = [1355.09 512.496   1];
angle_ranges_hypo2_our = [9.96648, 10.0231];

epipoles_hypo1_wacv = [1648.99 2758.96      1 ];
angle_ranges_hypo1_wacv = [64.8199, 64.8479];
epipoles_hypo2_wacv = [1355.09 512.496    1];
angle_ranges_hypo2_wacv = [9.01614, 9.07257];


mag = 0.1;
image_width = 800;
image_height = 800;

figure;
img_path = sprintf('/gpfs/data/bkimia/Datasets/ABC-NEF/0000%d/train_img/%d_colors.png', Object_Name, 30);
img = imread(img_path);
imshow(img); hold on;

% Plot original edges - cyan circles with tangent lines
plot(image_edges_our(:, 2), image_edges_our(:, 3), 'co', 'MarkerSize', 8, 'LineWidth', 2); hold on;
% plot([image_edges_our(:, 2) + mag*cos(image_edges_our(:, 4)), image_edges_our(:, 2) - mag*cos(image_edges_our(:, 4))], ...
%     [image_edges_our(:, 3) + mag*sin(image_edges_our(:, 4)), image_edges_our(:, 3) - mag*sin(image_edges_our(:, 4))], ...
%     'c-', 'LineWidth', 1.2); hold on;


plot(reprojection_our(:, 2), reprojection_our(:, 3), 'mo', 'MarkerSize', 8, 'LineWidth', 2); hold on;

plotLineOnImage(epipoles_hypo1_our, angle_ranges_hypo1_our(1), image_width, 'r-', 2);
plotLineOnImage(epipoles_hypo1_our, angle_ranges_hypo1_our(2), image_width, 'r-', 2);

% Hypothesis 2 - Blue solid
plotLineOnImage(epipoles_hypo2_our, angle_ranges_hypo2_our(1), image_width, 'b-', 2);
plotLineOnImage(epipoles_hypo2_our, angle_ranges_hypo2_our(2), image_width, 'b-', 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot WACV %%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot(image_edges_wacv(:, 2), image_edges_wacv(:, 3), 'cx', 'MarkerSize', 8, 'LineWidth', 2); hold on;
% Hypothesis 1 - Red dashed
plotLineOnImage(epipoles_hypo1_wacv, angle_ranges_hypo1_wacv(1), image_width, 'r--', 2);
plotLineOnImage(epipoles_hypo1_wacv, angle_ranges_hypo1_wacv(2), image_width, 'r--', 2);

% Hypothesis 2 - Blue dashed
plotLineOnImage(epipoles_hypo2_wacv, angle_ranges_hypo2_wacv(1), image_width, 'b--', 2);
plotLineOnImage(epipoles_hypo2_wacv, angle_ranges_hypo2_wacv(2), image_width, 'b--', 2);

plot(reprojection_wacv(:, 2), reprojection_wacv(:, 3), 'mx', 'MarkerSize', 8, 'LineWidth', 2); hold on;


function plotLineOnImage(epipole, angle_deg, image_width, type, lineWidth)
    if nargin < 5
        lineWidth = 2;
    end
    
    x = linspace(0, image_width, 1000);
    angle_rad = deg2rad(angle_deg);
    slope = tan(angle_rad);
    x1 = epipole(1);
    y1 = epipole(2);

    y = y1 + slope * (x - x1);
    plot(x, y, type, 'LineWidth', lineWidth);
    hold on;
end