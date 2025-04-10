clear;
clc;
close all;

%> Define the folder path containing the 3D edge output files
data_folder_name = 'outputs';
data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);

%> Specify the filename
file_name = '3D_edge_groups.txt';
file_path = fullfile(data_folder_path, file_name);

%> Load data: x, y, z, group_id
data = dlmread(file_path);
points = data(:, 1:3);
group_ids = data(:, 4);

%> Identify unique groups
unique_groups = unique(group_ids);
num_groups = length(unique_groups);

%> Define color map
colors = lines(36240);

%> Create figure
figure;
hold on;
view(3);

%> Plot each group in different color
for i = 1:num_groups
    gid = unique_groups(i);
    idx = (group_ids == gid);
    pts = points(idx, :);
    plot3(pts(:,1), pts(:,2), pts(:,3), '.', ...
          'Color', colors(i,:), ...
          'MarkerSize', 6, ...
          'DisplayName', sprintf('Group %d', gid));
end

%> Final plot settings
axis equal;
axis off;
set(gcf, 'color', 'w');
ax = gca;
ax.Clipping = "off";
title('3D Edges Colored by Group');
