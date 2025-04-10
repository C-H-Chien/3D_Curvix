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

%> Get all files matching the pattern
edge_files = dir(fullfile(data_folder_path, file_pattern));

%> Define a set of colors to be used for different files through 'lines' colormap
colors = lines(length(edge_files)); 

% Create a figure for plotting
figure;
hold on;

%> Loop through each file and plot its edges in 3D
for i = 1:length(edge_files)
    %> Read the current file
    current_file_path = fullfile(data_folder_path, edge_files(i).name);
    edges_file_read = fopen(current_file_path, 'r');
    disp(['Processing file: ', current_file_path]);

    %> Parse 3D edge data
    ldata = textscan(edges_file_read, '%f\t%f\t%f', 'CollectOutput', true);
    edges_3d = double(ldata{1,1});
    fclose(edges_file_read);

    %> Get the legend
    hypothesis_view1_index = extractBetween(edge_files(i).name, 'hypo1_', '_hypo2');
    hypothesis_view2_index = extractBetween(edge_files(i).name, 'hypo2_', '_t');
    show_legend = strcat("3D edges from hypothesis views ", hypothesis_view1_index, " and ", hypothesis_view2_index);

    % Define the target edges
    target_edge_blue = [0.589706, 0.304595, 0.67257];
    target_edge_green = [0.591346, 0.305229, 0.672385];

    % Compute distances and find matches
    distances_blue = abs(edges_3d - target_edge_blue);
    match_rows_blue = all(distances_blue < 0.001, 2);

    distances_green = abs(edges_3d - target_edge_green);
    match_rows_green = all(distances_green < 0.001, 2);

    % Plot matching blue points
    if any(match_rows_blue)
        disp('Blue target edge found!');
        plot3(edges_3d(match_rows_blue, 1), edges_3d(match_rows_blue, 2), edges_3d(match_rows_blue, 3), ...
              'Color', 'b', 'Marker', '.', 'LineStyle', 'none', ...
              'LineWidth', 0.1, 'DisplayName', strcat(show_legend, " (blue match)"));
    end

    % Plot matching green points
    if any(match_rows_green)
        disp('Green target edge found!');
        plot3(edges_3d(match_rows_green, 1), edges_3d(match_rows_green, 2), edges_3d(match_rows_green, 3), ...
              'Color', 'g', 'Marker', '.', 'LineStyle', 'none', ...
              'LineWidth', 0.1, 'DisplayName', strcat(show_legend, " (green match)"));
    end

    % Plot non-matching points
    non_match_rows = ~(match_rows_blue | match_rows_green);
    plot3(edges_3d(non_match_rows, 1), edges_3d(non_match_rows, 2), edges_3d(non_match_rows, 3), ...
          'Color', 'r', 'Marker', '.', 'LineStyle', 'none', ...
          'LineWidth', 0.1, 'DisplayName', show_legend);
end

%> Set the plot settings
axis equal;
axis off;
set(gcf, 'color', 'w');

%> Add a legend for each file
legend;
hold off;
