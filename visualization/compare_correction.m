clear;
close all;

% Dataset and image paths
Dataset_Path = "/gpfs/data/bkimia/Datasets/ABC-NEF/";
Object_Name = "00000006/";
Image_Path = fullfile(Dataset_Path, Object_Name, "train_img");
img_str = strcat(Image_Path, "/", string(42), "_colors.png");

% Load and display the image
img = imread(img_str);
imshow(img); 
hold on;

% Parameters for plotting
mag = 50; % Increased magnitude for visibility
original_point = [322.485, 364.499];
corrected_point = [322.498, 363.898];
tangent = 1.59197;

% Plot the original point with a visible marker
plot(original_point(1), original_point(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2); 
plot(corrected_point(1), corrected_point(2), 'bo', 'MarkerSize', 10, 'LineWidth', 2); 
% Plot the tangent line
plot([original_point(1) + mag * cos(tangent), original_point(1) - mag * cos(tangent)], ...
     [original_point(2) + mag * sin(tangent), original_point(2) - mag * sin(tangent)], ...
     'r', 'LineWidth', 2); 
hold on;

% Epipolar line parameters (a, b, c)
a = 0.992619;
b = -1;
c = 43.7804;

% Image dimensions
[img_height, img_width, ~] = size(img);

% Calculate two points on the epipolar line within the image boundaries
x1 = 1; % Start at the left edge of the image
y1 = -(a * x1 + c) / b; % Solve for y when x = 1

x2 = img_width; % End at the right edge of the image
y2 = -(a * x2 + c) / b; % Solve for y when x = img_width

% Clip the points to the image boundaries
if y1 < 1
    y1 = 1; % Ensure y1 is within the top boundary
    x1 = -(b * y1 + c) / a; % Recalculate x1 based on clipped y1
elseif y1 > img_height
    y1 = img_height; % Ensure y1 is within the bottom boundary
    x1 = -(b * y1 + c) / a; % Recalculate x1 based on clipped y1
end

if y2 < 1
    y2 = 1; % Ensure y2 is within the top boundary
    x2 = -(b * y2 + c) / a; % Recalculate x2 based on clipped y2
elseif y2 > img_height
    y2 = img_height; % Ensure y2 is within the bottom boundary
    x2 = -(b * y2 + c) / a; % Recalculate x2 based on clipped y2
end

% Plot the epipolar line
plot([x1, x2], [y1, y2], 'b', 'LineWidth', 2);

hold off;
