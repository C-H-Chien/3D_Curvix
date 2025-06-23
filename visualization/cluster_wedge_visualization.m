clear;
clc;
close all;

hypothesis2_edge_1 = [

];


edges_before_correction_1 = [
359.15    279.02   1.35685   7.69039
  359.218   279.517  -1.55337   7.51379
  359.194   280.012   -1.4987   7.49168
  359.154   280.509  -1.48486   7.52296


];

edges_after_correction_1 = [
354.101   304.327 -0.794065   8.82734
  359.631   274.748 -0.502888   70.0245
  358.979   278.235   1.35685   7.69039
  359.267   276.695  -1.55337   7.51379

];

cluster_avg_1 = [
  354.039   304.659   -0.8037   8.84477
  359.619   274.812 -0.394145   70.0245
  358.979   278.235   1.35685   7.69039
  359.267   276.695   1.58822   7.51379

];
epipoles_hypo2_1 = [-6931.8, 39276.8, 1];
angle_ranges_hypo2_1 = [100.588, 100.59];


hypothesis2_edge_2 = [];
edges_before_correction_2 = [

];

edges_after_correction_2 = [


];

cluster_avg_2 = [


];

epipoles_hypo2_2 = [];
angle_ranges_hypo2_2 = [];


mag = 0.5;
image_width = 800;
image_height = 800;
figure;
img_file = "/gpfs/data/bkimia/Datasets/ABC-NEF/00000006/train_img/48_colors.png";
imshow(img_file);
img2 = imread(img_file);
hold on;

% Create empty array to collect all plot handles
plot_handles_1 = [];
% epipolar line blue
h1 = plotLineOnImage(epipoles_hypo2_1, angle_ranges_hypo2_1(1), image_width, 'c--', 2);
h2 = plotLineOnImage(epipoles_hypo2_1, angle_ranges_hypo2_1(2), image_width, 'c--', 2);
plot_handles_1 = [plot_handles_1, h1];
% Plot edges before correction with red circles
h4 = plot(edges_before_correction_1(:,1), edges_before_correction_1(:,2), 'ro', 'MarkerSize', 10);
plot_handles_1 = [plot_handles_1, h4];
h5 = quiver(edges_before_correction_1(:,1), edges_before_correction_1(:,2), mag * cos(edges_before_correction_1(:,3)), mag * sin(edges_before_correction_1(:,3)), 0, 'r', 'LineWidth', 2);
plot_handles_1 = [plot_handles_1, h5];
% Plot edges after correction with blue circles
h6 = plot(edges_after_correction_1(:,1), edges_after_correction_1(:,2), 'bo', 'MarkerSize', 10);
plot_handles_1 = [plot_handles_1, h6];
u_after = mag * cos(edges_after_correction_1(:,3));
v_after = mag * sin(edges_after_correction_1(:,3));
h7 = quiver(edges_after_correction_1(:,1), edges_after_correction_1(:,2), u_after, v_after, 0, 'b', 'LineWidth', 2);
plot_handles_1 = [plot_handles_1, h7];
% Plot cluster centroids
h8 = plot(cluster_avg_1(:,1), cluster_avg_1(:,2), 'go', 'MarkerSize', 10);
plot_handles_1 = [plot_handles_1, h8];
u_before = mag * cos(cluster_avg_1(:,3));
v_before = mag * sin(cluster_avg_1(:,3));
h9 = quiver(cluster_avg_1(:,1), cluster_avg_1(:,2), u_before, v_before, 0, 'g', 'LineWidth', 2);
plot_handles_1 = [plot_handles_1, h9];
% h10 = plot(hypothesis2_edge_1(:, 1), hypothesis2_edge_1(:, 2), 'mo', 'MarkerSize', 10);
% plot_handles_1 = [plot_handles_1, h10];
% h11 = quiver(hypothesis2_edge_1(:, 1), hypothesis2_edge_1(:, 2), mag*cos(hypothesis2_edge_1(:, 3)), mag*sin(hypothesis2_edge_1(:, 3)), 0, 'm', 'LineWidth', 2);
% plot_handles_1 = [plot_handles_1, h11];


% % epipolar line blue
% h12 = plotLineOnImage(epipoles_hypo2_2, angle_ranges_hypo2_2(1), image_width, 'c-', 2);
% h12 = plotLineOnImage(epipoles_hypo2_2, angle_ranges_hypo2_2(2), image_width, 'c-', 2);
% % Plot edges before correction with red circles
% h13 = plot(edges_before_correction_2(:,1), edges_before_correction_2(:,2), 'ro', 'MarkerSize', 10);
% h14 = quiver(edges_before_correction_2(:,1), edges_before_correction_2(:,2), mag * cos(edges_before_correction_2(:,3)), mag * sin(edges_before_correction_2(:,3)), 0, 'r', 'LineWidth', 2);
% % Plot edges after correction with blue circles
% h15 = plot(edges_after_correction_2(:,1), edges_after_correction_2(:,2), 'bo', 'MarkerSize', 10);
% u_after = mag * cos(edges_after_correction_2(:,3));
% v_after = mag * sin(edges_after_correction_2(:,3));
% h16 = quiver(edges_after_correction_2(:,1), edges_after_correction_2(:,2), u_after, v_after, 0, 'b', 'LineWidth', 2);
% % Plot cluster centroids
% h17 = plot(cluster_avg_2(:,1), cluster_avg_2(:,2), 'go', 'MarkerSize', 10);
% u_before = mag * cos(cluster_avg_2(:,3));
% v_before = mag * sin(cluster_avg_2(:,3));
% h18 = quiver(cluster_avg_2(:,1), cluster_avg_2(:,2), u_before, v_before, 0, 'g', 'LineWidth', 2);
% h19 = plot(hypothesis2_edge_2(1), hypothesis2_edge_2(2), 'mo', 'MarkerSize', 10);
% h20 = quiver(hypothesis2_edge_2(1), hypothesis2_edge_2(2), mag*cos(hypothesis2_edge_2(3)), mag*sin(hypothesis2_edge_2(3)), 0, 'm', 'LineWidth', 2);
% 
% legend([h1, h4, h5, h6, h7, h8, h9, h10, h11, h12], {'Edge 1 epipolar wedge from hypothesis 1', 'Third-order edges', 'Third-order edge direction', 'Shifted third-order edge', 'Shifted third-order edge orientation', 'Cluster centroid', 'Cluster centroid orientation', 'Matched hypothesis 2 edge', 'Matched hypothesis 2 edge orientation', 'Edge 2 epipolar wedge from hypothesis 1'}, 'Location', 'northwest');

% legend([h1, h4, h5, h6, h7, h8, h9, h10, h11], {'Edge 1 epipolar wedge from hypothesis 1', 'Third-order edges', 'Third-order edge direction', 'Shifted third-order edge', 'Shifted third-order edge orientation', 'Cluster centroid', 'Cluster centroid orientation', 'Matched hypothesis 2 edge', 'Matched hypothesis 2 edge orientation'}, 'Location', 'northwest');
legend([h1, h4, h5, h6, h7, h8, h9], {'Edge 1 epipolar wedge from hypothesis 1', 'Third-order edges', 'Third-order edge direction', 'Shifted third-order edge', 'Shifted third-order edge orientation', 'Cluster centroid', 'Cluster centroid orientation'}, 'Location', 'northwest');

% Update the plotting functions to return handles
function h = plotLineOnImage(epipole, angle_deg, image_width, type, lineWidth)
    if nargin < 5
        lineWidth = 2;
    end
    
    x = linspace(0, image_width, 1000);
    angle_rad = deg2rad(angle_deg);
    slope = tan(angle_rad);
    x1 = epipole(1);
    y1 = epipole(2);
    y = y1 + slope * (x - x1);
    h = plot(x, y, type, 'LineWidth', lineWidth);
    hold on;
end

function h = plotGeneralLine(a, b, c, image_width, image_height, type, lineWidth)
    % This function plots a line in the form ax + by + c = 0
    % For a line in this form, we can rewrite as y = (-a/b)x - (c/b)
    % so the slope is -a/b and y-intercept is -c/b
    
    if nargin < 7
        lineWidth = 2;
    end
    
    % Create points that span the image
    x = linspace(0, image_width, 1000);
    
    % Calculate corresponding y values
    % y = (-a/b)x - (c/b)
    slope = -a/b;
    y_intercept = -c/b;
    y = slope * x + y_intercept;
    
    % Plot the line and return the handle
    h = plot(x, y, type, 'LineWidth', lineWidth);
    hold on;
end