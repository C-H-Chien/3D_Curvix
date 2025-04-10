%> mex function command:
%  $ mex Edge_Reconst/FastNViewTriangulation/examples/fast_multiview_triangulation_mex.cpp ...
%    Edge_Reconst/FastNViewTriangulation/src/NViewsCertifier.cpp Edge_Reconst/FastNViewTriangulation/src/NViewsClass.cpp ...
%    Edge_Reconst/FastNViewTriangulation/src/NViewsUtils.cpp Edge_Reconst/FastNViewTriangulation/utils/generatePointCloud.cpp ...
%    -I/gpfs/runtime/opt/eigen/3.3.2/include/eigen3

clear;
close all;

Dataset_Path = "/gpfs/data/bkimia/Datasets/ABC-NEF/";
Object_Name = "00000006/";
Image_Path = fullfile(Dataset_Path, Object_Name, "train_img");
Pose_Path = fullfile(Dataset_Path, Object_Name, "RnT");
Edges_Path = fullfile(Dataset_Path, Object_Name, "Edges");
skew_T = @(T)[0, -T(3,1), T(2,1); T(3,1), 0, -T(1,1); -T(2,1), T(1,1), 0];

data_folder_name = 'outputs';
data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);

%> Specify the common pattern in the file names
file_pattern = "3D_edges_*.txt";

K = [1111.11136542426,	0,	399.500000000000; 0,	1111.11136542426,	399.500000000000;  0,	0,	1];
K_inv = inv(K);
K_ = reshape(K', [1,9]);

%> Parsing object data
R_file = importdata(fullfile(Pose_Path, "R_matrix.txt"));
T_file = importdata(fullfile(Pose_Path, "T_matrix.txt"));

Edgels = cell(50,1);
R = cell(50,1);
T = cell(50,1);
for ei = 0:49
    Edgel_files{ei+1} = importdata(strcat(Edges_Path, "/Edge_", string(ei), "_t1.txt"));
    R{ei+1} = R_file(3*ei+1:3*ei+3, :);
    T{ei+1} = T_file(3*ei+1:3*ei+3);
end


%> Get all files matching the pattern
edges_3D_files = dir(fullfile(data_folder_path, file_pattern));
edges_3D = importdata(strcat(edges_3D_files(end).folder, "/", edges_3D_files(end).name));
disp(strcat(edges_3D_files(end).folder, "/", edges_3D_files(end).name));
edge_pair_final = importdata(fullfile(data_folder_path, "paired_edge_final.txt"));

H1_index = 47;
H2_index = 42;
image_indices = 0:1:49;
image_indices = image_indices';
image_indices([H1_index+1, H2_index+1], :) = [];
image_indices = [H1_index; H2_index; image_indices];

% Define the list of target edges
target_edges = [
    0.259948, 0.168784, 0.318437

];

% Prepare arrays to store results
results = struct('target_edge', [], 'captured_edge_pairs_img_indx', [], 'edges_2D_point', [], 'edges_2d_tangent', []);

% Loop through the target edges
for t = 1:size(target_edges, 1)
    % disp(t);
    target_edge = target_edges(t, :);
    target_index = find(abs(edges_3D(:,1) - target_edge(1)) < 0.00001 & ...
                        abs(edges_3D(:,2) - target_edge(2)) < 0.00001 & ...
                        abs(edges_3D(:,3) - target_edge(3)) < 0.00001);
    
    if (length(target_index) > 1)
        error("More than one target index for target edge [%f, %f, %f]!", target_edge);
    elseif isempty(target_index)
        warning("No target index found for target edge [%f, %f, %f]!", target_edge);
        continue;
    end

    % Get edge indices and corresponding image indices
    edge_indices = edge_pair_final(target_index, :);
    captured_edge_pairs_list_indx = find(edge_indices ~= -2);
    captured_edge_pairs_img_indx = image_indices(captured_edge_pairs_list_indx);
    captured_edge_pairs_edg_indices = edge_indices(captured_edge_pairs_list_indx);

    % Initialize variables for storing 2D points
    edges_2D_points = [];
    edges_3d_tangents = [];
    
    for i = 1:length(captured_edge_pairs_img_indx)
        % disp(i);
        img_indx = captured_edge_pairs_img_indx(i);
        edg_indx = captured_edge_pairs_edg_indices(i);
        edgels = Edgel_files{img_indx+1};
        edges_2D_point = [edgels(edg_indx+1, 1), edgels(edg_indx+1, 2)];
        edges_2D_points = [edges_2D_points; edges_2D_point];
        edges_3d_tangent = [edgels(edg_indx+1, 3)];
        edges_3d_tangents = [edges_3d_tangents; edges_3d_tangent];
    end

    % Store results
    results(t).target_edge = target_edge;
    results(t).captured_edge_pairs_img_indx = captured_edge_pairs_img_indx;
    results(t).edges_2D_point = edges_2D_points;
    results(t).edges_2d_tangent = edges_3d_tangents;
end

% Define colors for the 5 target edges
colors = ['r', 'g', 'b', 'm', 'c','r', 'g', 'b', 'm', 'c']; % Red, Green, Blue, Magenta, Cyan
marker_size = 5; 

figure;
mag = 1; 
for i = 1:length(results(1).captured_edge_pairs_img_indx)
    % Get the image index
    img_indx = results(1).captured_edge_pairs_img_indx(i);
    
    % Load the corresponding image
    img_path = fullfile(Image_Path, sprintf("%d_colors.png", img_indx));
    img = imread(img_path);

    subplot(ceil(length(results(1).captured_edge_pairs_img_indx) / 3), 3, i);
    imshow(img);
    hold on;
    title(sprintf("Image Index: %d", img_indx));
    
    fprintf('Image #%d\n', img_indx);
    for t = 1:length(results)
        points = results(t).edges_2D_point;
        tangents = results(t).edges_2d_tangent;
        if size(points, 1) >= i  

            plot(points(i, 1), points(i, 2),'o', 'Color', colors(t), 'MarkerSize', marker_size, 'LineWidth', 1);

            plot([points(i, 1)+mag*cos(tangents(i)), points(i, 1)-mag*cos(tangents(i))], ...
[points(i, 2)+mag*sin(tangents(i)), points(i, 2)-mag*sin(tangents(i))], ...
'Color', colors(t));
            if t==10

            %disp(tangents(i)); % Exclude from legend
                display(points(i,:));
            end
            %fprintf('(%f, %f)\n',points(i, 1), points(i, 2));
        end
    end
    hold off;
end

