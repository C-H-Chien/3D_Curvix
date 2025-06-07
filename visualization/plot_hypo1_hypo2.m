clear;
close all;

% Define dataset and image paths
Dataset_Path = "/gpfs/data/bkimia/Datasets/ABC-NEF/";
Object_Name = 8605;
Image_Path = fullfile(Dataset_Path, Object_Name, "train_img");

hypo1_edges = [
    529 398.495
];

hypo2_edge_1 = [
    436.066 366.827
];

hypo2_edge_2 = [
    419.014  379.018
];


figure;
subplot(1,2,1);
img_path = sprintf('/gpfs/data/bkimia/Datasets/ABC-NEF/0000%d/train_img/%d_colors.png', Object_Name, 19);
img = imread(img_path);
imshow(img);hold on;
plot(hypo1_edges(:, 1), hypo1_edges(:, 2), 'ro', 'MarkerSize', 8, 'LineWidth', 2);

subplot(1,2,2);
img_path = sprintf('/gpfs/data/bkimia/Datasets/ABC-NEF/0000%d/train_img/%d_colors.png', Object_Name, 25);
img = imread(img_path);
imshow(img);hold on;
plot(hypo2_edge_1(:, 1), hypo2_edge_1(:, 2), 'bo', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'our method');hold on;
plot(hypo2_edge_2(:, 1), hypo2_edge_2(:, 2), 'bx', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'WACV');hold on;

legend('show');

