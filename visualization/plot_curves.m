%% Visualize 3D Curves from Connectivity Graph
%  This script visualizes the curves extracted from the connectivity graph
%  returned by the 3D Curvix code.
%  (c) LEMS, Brown University
%  Qiwu Zhang and Chiang-Heng Chien

%> Define the folder path
data_folder_name = 'outputs';
data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);
curve_file = fullfile(data_folder_path, 'final_curves.txt');

%> Read the data
fileID = fopen(curve_file, 'r');
if fileID == -1
    error('Could not open file: %s', curve_file);
end

%> Skip the header lines
header = fgetl(fileID);
header2 = fgetl(fileID);

%> Initialize data structures
curve_data = [];
curve_ids = [];
node_indices = [];
current_curve_id = -1;

%> Read line by line
line = fgetl(fileID);
while ischar(line)
    %> Skip empty lines
    if ~isempty(line)
        data = sscanf(line, '%f');
        if length(data) >= 8
            curve_id = data(1);
            node_idx = data(2);
            x = data(3);
            y = data(4);
            z = data(5);
            dir_x = data(6);
            dir_y = data(7);
            dir_z = data(8);
            
            curve_data = [curve_data; x, y, z, dir_x, dir_y, dir_z];
            curve_ids = [curve_ids; curve_id];
            node_indices = [node_indices; node_idx];
            
            if curve_id ~= current_curve_id
                current_curve_id = curve_id;
            end
        end
    end
    line = fgetl(fileID);
end
fclose(fileID);

%> Get unique curve IDs
unique_curves = unique(curve_ids);
num_curves = length(unique_curves);
fprintf('Found %d curves\n', num_curves);


figure(100);
hold on;

%> Define a fixed set of 20 distinct colors that will repeat
sz = [num_curves 3];
curve_colors = unifrnd(0,1,sz);
% n = 12;
% fixed_colors = jet(n);

%> Plot each curve
for i = num_curves:-1:1
    curve_idx = curve_ids == unique_curves(i);
    points = curve_data(curve_idx, 1:3);
    
    %> Use modulo to cycle through the 20 colors
    % color_idx = mod(i-1, n) + 1;
    color = curve_colors(i,:);

    %> Prune out very short curves. This can be disabled.
    if size(points, 1) < 30
        continue;
    end
    
    %> Reverse the sign of the z-dimension to be consistent with the 3D GT
    %  curves from the ABC-NEF dataset (ignore if other dataset is used)
    plot3(points(:, 1), points(:, 2), -points(:, 3), '-', 'LineWidth', 2, 'Color', color);
    % scatter3(points(:, 1), points(:, 2), -points(:, 3), 'Marker', '.', 'Color', color);
end
title('3D Curves from Connectivity Graph');
axis off;
axis equal;
view(3); 
set(gcf, 'color', 'w');
ax = gca;
ax.Clipping = "off";
