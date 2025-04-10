clear;
close all;
clc;

% Dataset Path
Dataset_Path = "/gpfs/data/bkimia/Datasets/ABC-NEF/";
Object_Name = "00000006/";
Image_Path = fullfile(Dataset_Path, Object_Name, "train_img/");

% Load images
image_file_1 = fullfile(Image_Path, "47_colors.png");
image_file_2 = fullfile(Image_Path, "23_colors.png");

img1 = imread(image_file_1);
img2 = imread(image_file_2);

% Define multiple sets of values
values = [
    261.851, 441.599, 437.417, 738.764, 274.394, 405.998, -2.24936 -1 1722.67, 1.6926 -1 -1.6101];

% Create a figure
figure;

% Display image 47
subplot(1, 2, 1);
imshow(img1);
hold on;
title('Hypothesis 1 (image 47)');
colors = ['r', 'b', 'm', 'c', 'y', 'g'];

for i = 1:size(values, 1)
    plot(values(i, 5), values(i, 6), strcat(colors(mod(i-1, length(colors))+1), 'o'), 'MarkerSize', 12, 'LineWidth', 3, 'DisplayName', ['Edge Set ' num2str(i)]);
end
legend('Location', 'Best');
hold off;

% Display image 42 with points and lines
subplot(1, 2, 2);
imshow(img2);
hold on;
title('Hypothesis 2 (image 23)');

for i = 1:size(values, 1)
    plot(values(i, 1), values(i, 2), strcat(colors(mod(i-1, length(colors))+1), 'o'), 'MarkerSize', 5, 'LineWidth', 3, 'DisplayName', ['Original Edge Set ' num2str(i)]);
    plot(values(i, 3), values(i, 4), strcat(colors(mod(i-1, length(colors))+1), 'x'), 'MarkerSize', 5, 'LineWidth', 3, 'DisplayName', ['Epipolar Corrected Edge Set ' num2str(i)]);
    
    x_vals = linspace(0, size(img2, 2), 100);
    y_vals_epipolar = (-values(i, 7) * x_vals - values(i, 9)) / values(i, 8);
    plot(x_vals, y_vals_epipolar, strcat(colors(mod(i-1, length(colors))+1), '-'), 'LineWidth', 1, 'DisplayName', ['Epipolar Line Set ' num2str(i)]);
    
    y_vals_tangent = (-values(i, 10) * x_vals - values(i, 12)) / values(i, 11);
    plot(x_vals, y_vals_tangent, strcat(colors(mod(i-1, length(colors))+1), '--'), 'LineWidth', 1, 'DisplayName', ['Tangent Line Set ' num2str(i)]);
end

legend('Location', 'Best');
hold off;
