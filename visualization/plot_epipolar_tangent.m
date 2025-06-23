clear;
close all;

% === Settings ===
validation_view_index = 48;
Dataset_Path = "/gpfs/data/bkimia/Datasets/ABC-NEF/";
Object_Name = "00000006/";
Image_Path = fullfile(Dataset_Path, Object_Name, "train_img/");
img_filename = fullfile(Image_Path, sprintf('%d_colors.png', validation_view_index));
img = imread(img_filename);
mag = 48;

% === Line and point data ===
% Epipolar line: ax + by + c = 0 format
epipolar_a = -5.34903;
epipolar_b = -1;
epipolar_c = 2198.43;

% Tangent line: ax + by + c = 0 format  
tangent_a = 4.60262;
tangent_b = -1;
tangent_c = -1374.01;

% Points
corrected_x = 358.979;
corrected_y = 278.235;
edge_x = 359.15;
edge_y = 279.02;


% === Display the image ===
figure('Name', 'Image Analysis with Epipolar and Tangent Lines', 'Position', [100, 100, 1200, 800]);
imshow(img);
hold on;

% Get image dimensions for line plotting
[img_height, img_width, ~] = size(img);

% === Plot Epipolar Line ===
% For line ax + by + c = 0, solve for y at x boundaries
% y = -(ax + c)/b
x_range = [1, img_width];
if epipolar_b ~= 0
    y_epipolar = -(epipolar_a * x_range + epipolar_c) / epipolar_b;
    % Clip to image boundaries
    y_epipolar = max(1, min(img_height, y_epipolar));
    plot(x_range, y_epipolar, 'r-', 'LineWidth', 2, 'DisplayName', 'Epipolar Line');
end

% === Plot Tangent Line ===
if tangent_b ~= 0
    y_tangent = -(tangent_a * x_range + tangent_c) / tangent_b;
    % Clip to image boundaries
    y_tangent = max(1, min(img_height, y_tangent));
    plot(x_range, y_tangent, 'g-', 'LineWidth', 2, 'DisplayName', 'Tangent Line');
end

% === Plot Points ===
% Corrected location
plot(corrected_x, corrected_y, 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'blue', ...
     'DisplayName', sprintf('Corrected Location (%.3f, %.3f)', corrected_x, corrected_y));

% Edge point
plot(edge_x, edge_y, 'mo', 'MarkerSize', 8, 'MarkerFaceColor', 'magenta', ...
     'DisplayName', sprintf('Edge (%.2f, %.2f)', edge_x, edge_y));

% === Add annotations ===
% Annotate points
text(corrected_x + 10, corrected_y - 10, sprintf('Corrected\n(%.3f, %.3f)', corrected_x, corrected_y), ...
     'Color', 'blue', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', 'white');

text(edge_x + 10, edge_y + 10, sprintf('Edge\n(%.2f, %.2f)\nVals: %.3f, %.3f', ...
     edge_x, edge_y, edge_val1, edge_val2), ...
     'Color', 'magenta', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', 'white');

% === Add line equations as text ===
text(50, 50, sprintf('Epipolar: %.5fx + %.0fy + %.2f = 0', epipolar_a, epipolar_b, epipolar_c), ...
     'Color', 'red', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', 'white');

text(50, 80, sprintf('Tangent: %.5fx + %.0fy + %.2f = 0', tangent_a, tangent_b, tangent_c), ...
     'Color', 'green', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', 'white');

% === Formatting ===
title(sprintf('View %d - Epipolar and Tangent Line Analysis', validation_view_index), ...
      'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast');
grid on;
axis on;

% === Optional: Zoom to region of interest ===
% Uncomment to zoom around the points
% xlim([corrected_x - 100, corrected_x + 100]);
% ylim([corrected_y - 100, corrected_y + 100]);

% === Display information ===
fprintf('=== Analysis Results ===\n');
fprintf('Image: %s\n', img_filename);
fprintf('Epipolar line: %.5fx + %.0fy + %.2f = 0\n', epipolar_a, epipolar_b, epipolar_c);
fprintf('Tangent line: %.5fx + %.0fy + %.2f = 0\n', tangent_a, tangent_b, tangent_c);
fprintf('Corrected location: (%.3f, %.3f)\n', corrected_x, corrected_y);
fprintf('Edge location: (%.2f, %.2f) with values %.5f, %.5f\n', edge_x, edge_y, edge_val1, edge_val2);

hold off;