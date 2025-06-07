clear;
clc;
close all;

hypothesis2_edge_1 = [

];


edges_before_correction_1 = [
319.024   316.045 -0.485799    64.066
  319.618   315.726 -0.479939   63.6614
  319.425   315.857 -0.483381   63.8713
  320.016    315.53 -0.473807   64.0271
  320.097    320.69 -0.459942   11.4695
  319.905   320.818 -0.470069   11.4579
  320.504   320.509 -0.439517   11.5429
  321.102   320.219  -0.42651   11.5691
  320.919   320.331  -0.42905   11.5527
  357.888   510.672  0.576586   74.1444
  358.097    510.85  0.576725   74.2027
  358.464   511.056  0.573958   74.4344
  359.036   511.444  0.564635   74.5455

];

edges_after_correction_1 = [
319.584    315.75 -0.485799    64.066
   319.58   315.733 -0.479939   63.6614
  319.598   315.823 -0.483381   63.8713
  319.584   315.752 -0.473807   64.0271
  320.524   320.478 -0.459942   11.4695
  320.528   320.502 -0.470069   11.4579
  320.529   320.504 -0.439517   11.5429
  320.524   320.482  -0.42651   11.5691
   320.53   320.509  -0.42905   11.5527
  358.395   511.001  0.576586   74.1444
  358.355   510.799  0.576725   74.2027
  358.408   511.067  0.573958   74.4344
  358.403   511.043  0.564635   74.5455
];

cluster_avg_1 = [
319.587   315.764 -0.480731   63.9065
  319.587   315.764 -0.480731   63.9065
  319.587   315.764 -0.480731   63.9065
  319.587   315.764 -0.480731   63.9065
320.527   320.495 -0.445017   11.5184
  320.527   320.495 -0.445017   11.5184
320.527   320.495 -0.445017   11.5184
  320.527   320.495 -0.445017   11.5184
   358.39   510.978  0.572976   74.3317
   358.39   510.978  0.572976   74.3317
   358.39   510.978  0.572976   74.3317
   358.39   510.978  0.572976   74.3317
];
epipoles_hypo2_1 = [600.799 1730.49 1];
angle_ranges_hypo2_1 = [78.7278 78.7872];


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
img_file = "/gpfs/data/bkimia/Datasets/ABC-NEF/00000006/train_img/28_colors.png";
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

%legend([h1, h4, h5, h6, h7, h8, h9, h10, h11], {'Edge 1 epipolar wedge from hypothesis 1', 'Third-order edges', 'Third-order edge direction', 'Shifted third-order edge', 'Shifted third-order edge orientation', 'Cluster centroid', 'Cluster centroid orientation', 'Matched hypothesis 2 edge', 'Matched hypothesis 2 edge orientation'}, 'Location', 'northwest');
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