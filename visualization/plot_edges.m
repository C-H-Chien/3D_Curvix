clear;
close all;

%> Define the folder path containing the 3D edge and tangent output files
data_folder_name = 'outputs';
data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);

%> Edge view folder for observed edge points
observed_edge_folder = '/gpfs/data/bkimia/zqiwu/3D/3D_Edge_Sketch_and_Grouping/datasets/ABC-NEF/00000006/Edges/';

views = [2];

edges_1 = [274.736 448.018 -1.50485 70.7549];
edges_2 = [274.782 450.575 1.23129 67.2937];

mag = 5;


for i = 1:length(views)
    view_id = views(i);
    
    %> Construct the file path for the corresponding image
    img_path = sprintf('/gpfs/data/bkimia/Datasets/ABC-NEF/00000006/train_img/%d_colors.png', view_id);
    img = imread(img_path);

    figure;
    imshow(img); hold on;
    title(sprintf('Projected and Observed Edges on View %d', view_id));

    %> Load and plot observed 2D edge points (1st and 2nd columns only)
    observed_file = sprintf('Edge_%d_t1.txt', view_id);
    observed_path = fullfile(observed_edge_folder, observed_file);
    if isfile(observed_path)
        observed_edges = dlmread(observed_path);
        
        points = observed_edges(:, 1:2);
        tangents = observed_edges(:, 3);  % Orientation (assumed in radians)
        
        % Plot observed points
        plot(points(:,1), points(:,2), 'go', 'MarkerSize', 3, 'DisplayName', 'Observed Edges'); hold on;
        quiver(points(:,1), points(:,2), mag*cos(tangents), mag*sin(tangents),'g-', 'LineWidth', 1); hold on;
    else
        warning('Observed edge file not found: %s', observed_path);
    end
end

plot_handles = [];
h1 = plot(edges_1(:,1),  edges_1(:,2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
plot_handles = [plot_handles, h1];
u_after = mag * cos(edges_1(:,3));
v_after = mag * sin(edges_1(:,3));
quiver(edges_1(:,1), edges_1(:,2), u_after, v_after, 0, 'r', 'LineWidth', 2);hold on;

h2 = plot(edges_2(:,1),  edges_2(:,2), 'rx', 'LineWidth', 2, 'MarkerSize', 10);
plot_handles = [plot_handles, h2];
u_after = mag * cos(edges_2(:,3));
v_after = mag * sin(edges_2(:,3));
quiver(edges_2(:,1), edges_2(:,2), u_after, v_after, 0, 'r', 'LineWidth', 2);

legend([h1, h2], {'Edge 1', 'Edge 2'}, 'Location', 'northwest');


