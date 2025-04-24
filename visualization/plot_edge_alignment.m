clear; clc; close all;

%> Define the folder path containing the 3D edge output files
data_folder_name = 'outputs';
data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);
before_alignment_path = fullfile(data_folder_path, "test_3D_edges_before_smoothing.txt");
after_alignment_path = fullfile(data_folder_path, "test_3D_edges_after_smoothing.txt");
before_alignment = importdata(before_alignment_path);
after_alignment = importdata(after_alignment_path);

num_of_points = size(before_alignment, 1);
num_of_iters = size(after_alignment, 1) / num_of_points;
color_code = ["r", "b", "g", "c", "m", "y", "k"];
mag = 0.2;
figure;
for i = 1:num_of_points
    plot3(before_alignment(i,1), before_alignment(i,2), before_alignment(i,3), 'Color', color_code(i), 'Marker', 'o'); hold on;
    line([before_alignment(i,1) + mag*before_alignment(i,4), before_alignment(i,1) - mag*before_alignment(i,4)], ...
         [before_alignment(i,2) + mag*before_alignment(i,5), before_alignment(i,2) - mag*before_alignment(i,5)], ...
         [before_alignment(i,3) + mag*before_alignment(i,6), before_alignment(i,3) - mag*before_alignment(i,6)], ...
         'Color', color_code(i)); hold on;
end

rep_colors = repmat(color_code(1:num_of_points)', [num_of_iters, 1]);

for i = 1:size(after_alignment, 1)
    plot3(after_alignment(i,1), after_alignment(i,2), after_alignment(i,3), 'Color', rep_colors(i), 'Marker', 'o'); hold on;
    line([after_alignment(i,1) + mag*after_alignment(i,4), after_alignment(i,1) - mag*after_alignment(i,4)], ...
         [after_alignment(i,2) + mag*after_alignment(i,5), after_alignment(i,2) - mag*after_alignment(i,5)], ...
         [after_alignment(i,3) + mag*after_alignment(i,6), after_alignment(i,3) - mag*after_alignment(i,6)], ...
         'Color', rep_colors(i)); hold on;
end

direction_vector = after_alignment(end-1,1:3) - after_alignment(end-2,1:3);
t = linspace(-2, 1, 1000);
line_points = zeros(length(t), 3);
for i = 1:length(t)
    line_points(i,:) = after_alignment(end-1,1:3) + t(i) * direction_vector;
end
plot3(line_points(:,1), line_points(:,2), line_points(:,3), 'k--');

%> Set the plot settings
axis equal;
% axis off;
set(gcf, 'color', 'w');
ax = gca;
ax.Clipping = "off";

hold off;

