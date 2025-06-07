clc; clear; close all;

data_folder_name = 'outputs';
data_arxiv_folder_name = 'outputs_arXiv';
data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);
data_arxiv_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_arxiv_folder_name);

%> Load the 3D edge graph
%Edge_Graph = importdata(fullfile(data_folder_path, "3D_edge_graph.txt"));
%Edge_Graph = importdata(fullfile(data_folder_path, "/pruned_graph_aligned.txt"));
Edge_Graph = importdata(fullfile(data_folder_path, "/3D_edge_pruned_graph_by_proj.txt"));
Edges_3D_Graph_Locations = Edge_Graph(:,1:9);

% x_min = 0.33;
% x_max = 0.58;
% y_min = 0.017;
% y_max = 0.02;
% z_min = 0.991;
% z_max = 1.1;
% index = find(Edges_3D_Graph_Locations(:,1) < x_max & Edges_3D_Graph_Locations(:,1) > x_min & ...
%              Edges_3D_Graph_Locations(:,2) < y_max & Edges_3D_Graph_Locations(:,2) > y_min & ...
%              Edges_3D_Graph_Locations(:,3) < z_max & Edges_3D_Graph_Locations(:,3) > z_min);
index = 1:size(Edge_Graph,1);

figure;
graph_links_X = [Edge_Graph(index,1), Edge_Graph(index,4)]';
graph_links_Y = [Edge_Graph(index,2), Edge_Graph(index,5)]';
graph_links_Z = [Edge_Graph(index,3), Edge_Graph(index,6)]';
line(graph_links_X, graph_links_Y, graph_links_Z, 'Color', 'g');
hold on;

%> ----------------------------------------------------------------------

%> Load the 3D edge locations and orientations
%> (1) Specify the common patterns in the file names
edge_file_pattern = "3D_edges_ABC*.txt";
tangent_file_pattern = "3D_tangents_*.txt";
%> (2) Get all files matching the patterns
edge_files = dir(fullfile(data_folder_path, edge_file_pattern));
tangent_files = dir(fullfile(data_folder_path, tangent_file_pattern));


mag = 0.01; 
view(3);
axis equal;
axis off;
set(gcf, 'color', 'w');
ax = gca;
ax.Clipping = "off";

plot3(Edges_3D_Graph_Locations(:,1), Edges_3D_Graph_Locations(:,2), Edges_3D_Graph_Locations(:,3), ...
  'Color', 'b', 'Marker', '.', 'LineStyle', 'none', ...
  'LineWidth', 0.1, 'MarkerSize', 5);
hold on;

plot3(Edges_3D_Graph_Locations(:,4), Edges_3D_Graph_Locations(:,5), Edges_3D_Graph_Locations(:,6), ...
  'Color', 'b', 'Marker', '.', 'LineStyle', 'none', ...
  'LineWidth', 0.1, 'MarkerSize', 5);
hold on;

quiver3(Edges_3D_Graph_Locations(:,1), Edges_3D_Graph_Locations(:,2), Edges_3D_Graph_Locations(:,3), mag*Edges_3D_Graph_Locations(:,7), ...
    mag*Edges_3D_Graph_Locations(:,8), mag*Edges_3D_Graph_Locations(:,9),0, 'Color', 'k', 'LineWidth', 0.1, 'MaxHeadSize', 0.1);
%> Loop through each file and plot its edges and tangents in 3D
% for i = 1:length(edge_files)
% 
%     edges_3d = importdata(fullfile(data_folder_path, edge_files(i).name));
%     tangents_3d = importdata(fullfile(data_folder_path, tangent_files(i).name));
% 
%     plot3(edges_3d(:, 1),edges_3d(:, 2),edges_3d(:, 3), 'Color', 'b', 'Marker', '.', 'LineStyle', 'none','LineWidth', 0.2, 'MarkerSize', 10);hold on;
% 
%     quiver3(edges_3d(:, 1),edges_3d(:, 2),edges_3d(:, 3), mag*tangents_3d(:,1), mag*tangents_3d(:,2), mag*tangents_3d(:,3),0, 'Color', 'k', 'LineWidth', 0.5, 'MaxHeadSize', 0.5)
%     t_lines_X = [edges_3d(:,1) + mag*tangents_3d(:,1), edges_3d(:,1) - mag*tangents_3d(:,1)]';
%     t_lines_Y = [edges_3d(:,2) + mag*tangents_3d(:,2), edges_3d(:,2) - mag*tangents_3d(:,2)]';
%     t_lines_Z = [edges_3d(:,3) + mag*tangents_3d(:,3), edges_3d(:,3) - mag*tangents_3d(:,3)]';
%     line(t_lines_X, t_lines_Y, t_lines_Z, 'Color', 'b');
%     hold on;
% end

% plot3(0.497567, 1.00021, 0.431194, 'Color', 'r', 'Marker', '.', 'LineStyle', 'none', ...
%           'LineWidth', 0.2, 'MarkerSize', 10);hold on;
% plot3(0.498778, 1.0002, 0.431588, 'Color', 'r', 'Marker', '.', 'LineStyle', 'none', ...
%           'LineWidth', 0.2, 'MarkerSize', 10);hold on;


% plot3(0.0675822,   0.69912,  0.352804, 'Color', 'r', 'Marker', '.', 'LineStyle', 'none', ...
%           'LineWidth', 0.2, 'MarkerSize', 10);hold on;
% plot3(0.0674756,  0.699101,  0.353034, 'Color', 'r', 'Marker', '.', 'LineStyle', 'none', ...
%           'LineWidth', 0.2, 'MarkerSize', 10);hold on;





