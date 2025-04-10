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
file_pattern = "3D_edges_*.txt";
tangent_file_pattern = "original_tangents.txt";

%> Get all files matching the pattern
edge_files = dir(fullfile(data_folder_path, file_pattern));
tangent_files = dir(fullfile(data_folder_path, tangent_file_pattern));

% Scalar for tangent vectors
mag = 0.0025;

%> Define a set of colors to be used for different files through 'jet' colormap
numColors = 10;
colormapMatrix = jet(numColors);

% Create a figure for plotting
figure;
hold on;

%> Loop through each file and plot its edges in 3D
for i = 1:length(edge_files)
    %> Read the current file
    current_file_path = fullfile(data_folder_path, edge_files(i).name);
    edges_file_read = fopen(current_file_path, 'r');
    disp(current_file_path);

    %> Parse 3D edge data
    ldata = textscan(edges_file_read, '%f\t%f\t%f\t%f', 'CollectOutput', true);
    edges_3d = double(ldata{1,1});
    fclose(edges_file_read);

    %> Find the corresponding tangent file
    current_tangent_file_path = fullfile(data_folder_path, tangent_file_pattern);
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


    %> Get the legend
    hypothesis_view1_index = extractBetween(edge_files(i).name, 'hypo1_', '_hypo2');
    hypothesis_view2_index = extractBetween(edge_files(i).name, 'hypo2_', '_t');
    show_legend = strcat("3D edges from hypothesis views ", hypothesis_view1_index, " and ", hypothesis_view2_index);

    %> Loop through each line in the file and plot it separately
    for j = 1:size(edges_3d, 1)
        colorIndex = edges_3d(j, 4);
        r = mod(colorIndex - 1, numColors) + 1;


        %> Plot the edge as a line in 3D
        % plot3(edges_3d(j, 1), edges_3d(j, 2), edges_3d(j, 3), ...
        %       'Color', colormapMatrix(r, :), 'Marker', '.', 'LineStyle', 'none', ...
        %       'LineWidth', 0.1, 'MarkerSize', 3);
        plot3(edges_3d(j, 1), edges_3d(j, 2), edges_3d(j, 3), ...
              'Color', 'b', 'Marker', '.', 'LineStyle', 'none', ...
              'LineWidth', 0.1, 'MarkerSize', 3);

        P = edges_3d(j, :)'; % Current edge point
        T_v = tangents_3d(j, :)'; % Corresponding tangent vector

        % Plot tangent vector as a line originating from the point, no legend
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

hold off;
