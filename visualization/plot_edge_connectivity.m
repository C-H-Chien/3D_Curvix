clear; close all;

%> Define the folder path containing the 3D edge output files
data_folder_name = 'outputs';
data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);

% Load the data files
after_alignment_path = fullfile(data_folder_path, "test_3D_edges_before_connectivity_graph.txt");
after_alignment = importdata(after_alignment_path);
c_graph_path = fullfile(data_folder_path, "test_3D_edges_after_connectivity_graph.txt");
connectivity_graph = importdata(c_graph_path);
target_left_right_path = fullfile(data_folder_path, "test_3D_edge_left_and_right.txt");
taget_left_right = importdata(target_left_right_path);

% Calculate number of points and verify iteration is valid
num_of_points = size(after_alignment, 1);

% Set magnitude scaling factors
mag_orientation = 0.0005;  % for orientation vectors
mag_force = 0.5;        % for force vectors

% Create figure
figure(1);
subplot(1,2,1);

% Get the indices for the requested iteration
start_idx = 1;

% Plot each point for the given iteration
for i = 0:num_of_points-1
    % % Get correct color index with wraparound
    % color_idx = mod(i, size(color_code, 1)) + 1;
    
    % Get the point index in the after_alignment array
    point_idx = start_idx + i;
    
    % Get the current point location
    point_loc = after_alignment(point_idx,1:3);
    
    % Get the orientation vector
    orientation_vec = after_alignment(point_idx,4:6);
    
    % Plot the point location
    plot3(point_loc(1), point_loc(2), point_loc(3), ...
        'Color', 'b', 'Marker', 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'b'); 
    hold on;
    
    % Plot orientation vector as a line through the point
    quiver3(point_loc(1), point_loc(2), point_loc(3), ...
            mag_orientation*orientation_vec(1), mag_orientation*orientation_vec(2), mag_orientation*orientation_vec(3), ...
            0, 'LineWidth', 1, 'Color', 'b', 'MaxHeadSize', 0.5);
    % line([point_loc(1) + mag_orientation*orientation_vec(1), point_loc(1) - mag_orientation*orientation_vec(1)], ...
    %      [point_loc(2) + mag_orientation*orientation_vec(2), point_loc(2) - mag_orientation*orientation_vec(2)], ...
    %      [point_loc(3) + mag_orientation*orientation_vec(3), point_loc(3) - mag_orientation*orientation_vec(3)], ...
    %      'Color', color_code(color_idx,:), 'LineWidth', 1); 
end

% Set plot properties
axis equal;
title({'Edge Locations and Orientations', 'Before Constructing Connectivity Graph'});
xlabel('X');
ylabel('Y');
zlabel('Z');
grid on;

% Set the background color to white
set(gcf, 'color', 'w');
ax = gca;
ax.Clipping = "off";
hold off;

pause(0.5);
subplot(1,2,2);
start_idx = 1;

% Plot each point for the given iteration
for i = 0:num_of_points-1
    
    point_idx = start_idx + i;
    point_loc = connectivity_graph(point_idx,1:3);
    orientation_vec = connectivity_graph(point_idx,4:6);
    
    % Plot the point location
    plot3(point_loc(1), point_loc(2), point_loc(3), ...
        'Color', 'b', 'Marker', 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'b'); 
    hold on;
    
    % Plot orientation vector as a line through the point
    quiver3(point_loc(1), point_loc(2), point_loc(3), ...
            mag_orientation*orientation_vec(1), mag_orientation*orientation_vec(2), mag_orientation*orientation_vec(3), ...
            0, 'LineWidth', 1, 'Color', 'b', 'MaxHeadSize', 0.5);
end

hold on;
%> target edge
color_code = ["g", "m", "m"];
for i = 1:3
    plot3(taget_left_right(i,1), taget_left_right(i,2), taget_left_right(i,3), ...
        'Color', color_code(i), 'Marker', 'o', 'MarkerSize', 5, 'MarkerFaceColor', color_code(i));
    hold on;
    quiver3(taget_left_right(i,1), taget_left_right(i,2), taget_left_right(i,3), ...
            mag_orientation*taget_left_right(i,4), mag_orientation*taget_left_right(i,5), mag_orientation*taget_left_right(i,6), ...
            0, 'LineWidth', 1, 'Color', color_code(i), 'MaxHeadSize', 0.5);
    hold on;
end

% Set plot properties
axis equal;
title({'Edge Locations and Orientations', 'After Constructing Connectivity Graph'});
xlabel('X');
ylabel('Y');
zlabel('Z');
grid on;

% Set the background color to white
set(gcf, 'color', 'w');
ax = gca;
ax.Clipping = "off";
hold off;