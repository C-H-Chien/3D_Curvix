clc; clear; close all;

data_folder_name = 'outputs';
data_arxiv_folder_name = 'outputs_arXiv';
data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);
data_arxiv_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_arxiv_folder_name);

%> Load the 3D edge graph
% Edge_Graph = importdata(fullfile(data_arxiv_folder_path, "3D_edge_pruned_graph_lambda1_p5_lambda2_p5.txt"));
Edge_Graph = importdata(fullfile(data_folder_path, "3D_edge_pruned_graph_by_proj.txt"));
Edges_3D_Graph_Locations = Edge_Graph(:,1:3);

% x_min = 0.067;
% x_max = 0.068;
% y_min = 0.57;
% y_max = 0.62;
% z_min = 0.34;
% z_max = 0.351;
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
edge_file_pattern = "3D_edges_*.txt";
tangent_file_pattern = "3D_tangents_*.txt";
%> (2) Get all files matching the patterns
edge_files = dir(fullfile(data_folder_path, edge_file_pattern));
tangent_files = dir(fullfile(data_folder_path, tangent_file_pattern));


mag = 0.0025; 

%> Loop through each file and plot its edges and tangents in 3D
for i = 1:length(edge_files)
    
    edges_3d = importdata(fullfile(data_folder_path, edge_files(i).name));
    tangents_3d = importdata(fullfile(data_folder_path, tangent_files(i).name));

    plot3(edges_3d(:,1), edges_3d(:,2), edges_3d(:,3), ...
          'Color', 'b', 'Marker', '.', 'LineStyle', 'none', ...
          'LineWidth', 0.2, 'MarkerSize', 5);
    hold on;

    % t_lines_X = [edges_3d(:,1) + mag*tangents_3d(:,1), edges_3d(:,1) - mag*tangents_3d(:,1)]';
    % t_lines_Y = [edges_3d(:,2) + mag*tangents_3d(:,2), edges_3d(:,2) - mag*tangents_3d(:,2)]';
    % t_lines_Z = [edges_3d(:,3) + mag*tangents_3d(:,3), edges_3d(:,3) - mag*tangents_3d(:,3)]';
    % line(t_lines_X, t_lines_Y, t_lines_Z, 'Color', 'b');
    % hold on;
end

view(3);
%> Set the plot settings
axis equal;
axis off;
set(gcf, 'color', 'w');
ax = gca;
ax.Clipping = "off";
hold off;
