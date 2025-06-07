clc; clear; close all;
data_folder_name = 'outputs';
data_arxiv_folder_name = 'outputs_arXiv';
data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);
data_arxiv_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_arxiv_folder_name);

%> Load the 3D edge graph
Edge_Graph = importdata(fullfile(data_folder_path, "/pruned_graph_aligned.txt"));


Edges_3D_Graph_Locations = Edge_Graph(:,1:3);

x_min = 0.93;
x_max = 0.94;
y_min = 0.25;
y_max = 0.26;
z_min = 0.36;
z_max = 0.5;
index = find(Edges_3D_Graph_Locations(:,1) < x_max & Edges_3D_Graph_Locations(:,1) > x_min & ...
             Edges_3D_Graph_Locations(:,2) < y_max & Edges_3D_Graph_Locations(:,2) > y_min & ...
             Edges_3D_Graph_Locations(:,3) < z_max & Edges_3D_Graph_Locations(:,3) > z_min);
% index = 1:size(Edge_Graph,1);
% Create a figure
figure;
graph_links_X = [Edge_Graph(index,1), Edge_Graph(index,4)]';
graph_links_Y = [Edge_Graph(index,2), Edge_Graph(index,5)]';
graph_links_Z = [Edge_Graph(index,3), Edge_Graph(index,6)]';
line(graph_links_X, graph_links_Y, graph_links_Z, 'Color', 'g');
hold on;

% Plot the edge start points as blue dots
plot3(Edges_3D_Graph_Locations(:,1), Edges_3D_Graph_Locations(:,2), Edges_3D_Graph_Locations(:,3), 'Color', 'b', 'Marker', '.', 'LineStyle', 'none', 'MarkerSize', 4);hold on;
plot3(0.93316, 0.250404, 0.463234, 'Color', 'r', 'Marker', '.', 'LineStyle', 'none', 'MarkerSize', 4)
plot3(0.933665, 0.250409, 0.460146, 'Color', 'r', 'Marker', '.', 'LineStyle', 'none', 'MarkerSize', 4)

% Configure the plot appearance
view(3);
axis equal;
axis off;
set(gcf, 'color', 'w');
ax = gca;
ax.Clipping = "off";

% Add a title and legend
% title('Edge Connections', 'FontSize', 14);
% legend('Connections', 'Edge Start Points', 'Edge End Points', 'Location', 'best');