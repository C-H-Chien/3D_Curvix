%> Automatically plot 3D edge files before and after smoothing
%
%> (c) LEMS, Brown University
%> chiang-heng chien
clear;
clc;
close all;

%> Define the folder path
data_folder_name = 'outputs';
data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);

%> File names
before_file = fullfile(data_folder_path, '3D_edges_before_smoothing.txt');
after_file  = fullfile(data_folder_path, '3D_edges_after_smoothing.txt');

%> Scalar for tangent vectors
mag = 0.0025;

%> Load before-smoothing edges
if exist(before_file, 'file')
    fid_before = fopen(before_file, 'r');
    before_data = textscan(fid_before, '%f %f %f', 'CollectOutput', true);
    edges_before = before_data{1};
    fclose(fid_before);
else
    error('File not found: %s', before_file);
end

%> Load after-smoothing edges
if exist(after_file, 'file')
    fid_after = fopen(after_file, 'r');
    after_data = textscan(fid_after, '%f %f %f', 'CollectOutput', true);
    edges_after = after_data{1};
    fclose(fid_after);
else
    error('File not found: %s', after_file);
end

%> Plot setup
figure;
hold on;
view(3);
axis equal;
axis off;
set(gcf, 'color', 'w');
ax = gca;
ax.Clipping = "off";

% > Plot before smoothing in red
plot3(edges_before(:,1), edges_before(:,2), edges_before(:,3), '.', ...
      'Color', [1 0.2 0.2], 'MarkerSize', 3, 'DisplayName', 'Before Smoothing');

%> Plot after smoothing in blue
% plot3(edges_after(:,1), edges_after(:,2), edges_after(:,3), '.', ...
%       'Color', [0.2 0.4 1], 'MarkerSize', 3, 'DisplayName', 'After Smoothing');

legend('Location', 'best');
title('3D Edges Before and After Smoothing');

hold off;
