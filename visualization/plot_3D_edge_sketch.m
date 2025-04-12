%> Automatically plot every 3D edge file in the folder with a different color
%
%> (c) LEMS, Brown University
%> chiang-heng chien
clear;
clc;
close all;

%> Define the folder path containing the 3D edge output files
data_folder_name = 'outputs';
data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);

%> Specify the common pattern in the file names
file_pattern = "grouped_mvt.txt";
tangent_file_pattern = "grouped_mvt_tangent.txt";
%> Get all files matching the pattern
edge_files = dir(fullfile(data_folder_path, file_pattern));
tangent_file = fullfile(data_folder_path, tangent_file_pattern);
%> Define a set of colors to be used for different files through 'lines' colormap
colors = lines(length(edge_files)); 

% Create a figure for plotting
figure;
hold on;

%> Scalar for tangent vectors
mag = 0.0025; 

%> Loop through each file and plot its edges in 3D
for i = 1:length(edge_files)
    %> Read the current 3D edge file
    current_file_path = fullfile(data_folder_path, edge_files(i).name);
    edges_file_read = fopen(current_file_path, 'r');
    disp(current_file_path);
    
    %> Read the corresponding tangent file
    if ~exist(tangent_file, 'file')
        disp(['Tangent file not found for: ', tangent_file]);
        continue;
    end
    tangent_file_read = fopen(tangent_file, 'r');
    disp(['Reading tangents: ', tangent_file]);
    tangent_data = textscan(tangent_file_read, '%f\t%f\t%f', 'CollectOutput', true);
    tangents_3d = double(tangent_data{1, 1});
    fclose(tangent_file_read);

    %> parse 3D edge data
    ldata = textscan(edges_file_read, '%f\t%f\t%f', 'CollectOutput', true);
    edges_3d = double(ldata{1,1});
    fclose(edges_file_read);
    
    %> Get the legend
    hypothesis_view1_index = extractBetween(edge_files(i).name, 'hypo1_', '_hypo2');
    hypothesis_view2_index = extractBetween(edge_files(i).name, 'hypo2_', '_t');
    show_legend = strcat("3D edges from hypothesis views ", hypothesis_view1_index, " and ", hypothesis_view2_index);

    %> Plot the edges using a different color for each file
    % plot3(edges_3d(:,1), edges_3d(:,2), edges_3d(:,3), ...
    %       'Color', colors(i, :), 'Marker', '.', 'LineStyle', 'none', ...
    %       'LineWidth', 0.1, 'MarkerSize', 3, 'DisplayName', show_legend);
    disp("Using plot3: ");
    plot3(edges_3d(:,1), edges_3d(:,2), edges_3d(:,3), '.', 'LineWidth', 0.1, 'MarkerSize', 3);
    
    % for j = 1:size(edges_3d, 1)
    %     P = edges_3d(j, :)'; % Current edge point
    %     T_v = tangents_3d(j, :)'; % Corresponding tangent vector
    % 
    %     % Plot tangent vector as a line originating from the point, no legend
    %     line([P(1) + mag*T_v(1), P(1) - mag*T_v(1)], ...
    %          [P(2) + mag*T_v(2), P(2) - mag*T_v(2)], ...
    %          [P(3) + mag*T_v(3), P(3) - mag*T_v(3)], ...
    %          'Color', 'k', 'LineWidth', 0.5, 'HandleVisibility', 'off'); % Exclude from legend
    % end
    view(3); % Set 3D perspective


end

%> Set the plot settings
axis equal;
axis off;
set(gcf, 'color', 'w');
ax = gca;
ax.Clipping = "off";

hold off;