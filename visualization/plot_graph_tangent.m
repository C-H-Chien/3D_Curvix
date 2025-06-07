data_folder_name = 'outputs';
data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);
%curve_file = fullfile(data_folder_path, 'curves_from_connectivity_graph.txt');
curve_file = fullfile(data_folder_path, 'aligned_orientation.txt');

%> Load before-smoothing edges
if exist(curve_file, 'file')
    fid_before = fopen(curve_file, 'r');
    before_data = textscan(fid_before, '%f %f %f %f %f %f %f %f', 'CollectOutput', true);
    edges_before = before_data{1};
    fclose(fid_before);
else
    error('File not found: %s', before_file);
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
mag = 0.005;

% > Plot before smoothing in red
plot3(edges_before(:,3), edges_before(:,4), edges_before(:,5), '.', ...
      'Color', [1 0.2 0.2], 'MarkerSize', 8, 'DisplayName', 'Before Smoothing');

quiver3(edges_before(:,3), edges_before(:,4), edges_before(:,5), mag*edges_before(:,6), mag*edges_before(:,7), mag*edges_before(:,8),0, 'Color', 'k', 'LineWidth', 0.5, 'MaxHeadSize', 0.5);