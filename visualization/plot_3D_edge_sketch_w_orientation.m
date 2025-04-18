%> Plot 3D edge sketch represented by 3D edge locations and orientations (3D tangents)
%
%> (c) LEMS, Brown University
%> chiang-heng chien

clear;
clc;
close all;

%> Define the folder path containing the 3D edge and tangent output files
data_folder_name = 'outputs';
data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);

%> Specify the common patterns in the file names
edge_file_pattern = "3D_edges_*.txt";
tangent_file_pattern = "3D_tangents_*.txt";

%> Get all files matching the patterns
edge_files = dir(fullfile(data_folder_path, edge_file_pattern));
tangent_files = dir(fullfile(data_folder_path, tangent_file_pattern));

%> Define a set of colors to be used for different files through 'lines' colormap
% colors = lines(length(edge_files)); 

% Scalar for tangent vectors
mag = 0.0025; 

% Create a figure for plotting
figure;
hold on;

%> Loop through each file and plot its edges and tangents in 3D
for i = 1:length(edge_files)
    %> Read the current edge file
    current_edge_file_path = fullfile(data_folder_path, edge_files(i).name);
    edge_file_read = fopen(current_edge_file_path, 'r');
    disp(['Reading edges: ', current_edge_file_path]);

    %> Parse 3D edge data
    edge_data = textscan(edge_file_read, '%f\t%f\t%f', 'CollectOutput', true);
    edges_3d = double(edge_data{1, 1});
    fclose(edge_file_read);

    %> Find the corresponding tangent file
    current_tangent_file_path = fullfile(data_folder_path, tangent_files(i).name);
    if ~exist(current_tangent_file_path, 'file')
        disp(['Tangent file not found for: ', current_edge_file_path]);
        continue;
    end

    %> Read the corresponding tangent file
    tangent_file_read = fopen(current_tangent_file_path, 'r');
    disp(['Reading tangents: ', current_tangent_file_path]);
    tangent_data = textscan(tangent_file_read, '%f\t%f\t%f', 'CollectOutput', true);
    tangents_3d = double(tangent_data{1, 1});
    fclose(tangent_file_read);

    %> Check for size mismatch between edges and tangents
    if size(edges_3d, 1) ~= size(tangents_3d, 1)
        warning(['Mismatch between number of edges and tangents in file: ', edge_files(i).name]);
        disp(['Number of edges: ', num2str(size(edges_3d, 1)), ...
              ', Number of tangents: ', num2str(size(tangents_3d, 1))]);
        continue;
    end

    %> Get the legend
    % hypothesis_view1_index = extractBetween(edge_files(i).name, 'hypo1_', '_hypo2');
    % hypothesis_view2_index = extractBetween(edge_files(i).name, 'hypo2_', '_t');
    % show_legend = strcat("3D edges from hypothesis views ", hypothesis_view1_index, " and ", hypothesis_view2_index);

    %> Plot the edges using a different color for each file
    % plot3(edges_3d(:,1), edges_3d(:,2), edges_3d(:,3), ...
          % 'Color', colors(i, :), 'Marker', '.', 'LineStyle', 'none', ...
          % 'LineWidth', 0.1, 'MarkerSize', 3, 'DisplayName', show_legend);
    plot3(edges_3d(:,1), edges_3d(:,2), edges_3d(:,3), ...
          'Color', 'b', 'Marker', '.', 'LineStyle', 'none', ...
          'LineWidth', 0.1, 'MarkerSize', 3);
    hold on;

    % t_lines_X = [edges_3d(:,1) + mag*tangents_3d(:,1), edges_3d(:,1) - mag*tangents_3d(:,1)]';
    % t_lines_Y = [edges_3d(:,2) + mag*tangents_3d(:,2), edges_3d(:,2) - mag*tangents_3d(:,2)]';
    % t_lines_Z = [edges_3d(:,3) + mag*tangents_3d(:,3), edges_3d(:,3) - mag*tangents_3d(:,3)]';
    % 
    % line(t_lines_X, t_lines_Y, t_lines_Z, 'Color', 'b');


    %> Plot the tangent vectors for each edge
    for j = 1:size(edges_3d, 1)
        P = edges_3d(j, :)'; % Current edge point
        T_v = tangents_3d(j, :)'; % Corresponding tangent vector

        line([P(1) + mag*T_v(1), P(1) - mag*T_v(1)], ...
             [P(2) + mag*T_v(2), P(2) - mag*T_v(2)], ...
             [P(3) + mag*T_v(3), P(3) - mag*T_v(3)], ...
             'Color', 'k', 'LineWidth', 0.5, 'HandleVisibility', 'off'); % Exclude from legend
    end
end

%> Set the plot settings
axis equal;
axis off;
set(gcf, 'color', 'w');
ax = gca;
ax.Clipping = "off";

%> Add a legend for each file
% legend;

hold off;
