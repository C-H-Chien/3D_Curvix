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

%> Let's plot 3D edges first
% figure(1);
% plot3(edges_3D(:,1), edges_3D(:,2), edges_3D(:,3), 'Color', 'r', 'Marker', '.', 'LineStyle', 'none')

<<<<<<< HEAD
H1_index = 47;
H2_index = 42;
=======
H1_index = 0;
H2_index = 40;
>>>>>>> e6c6edf843f3f019e8d3420d15a62864487d5a8f
image_indices = 0:1:49;
image_indices = image_indices';
image_indices([H1_index+1, H2_index+1], :) = [];
%image_indices(H2_index+1, :) = [];
image_indices = [H1_index; H2_index; image_indices];

<<<<<<< HEAD
target_edge = [0.710596, 0.116216, 0.351873];
target_index = find(abs(edges_3D(:,1)-target_edge(1))<0.00001 & abs(edges_3D(:,2)-target_edge(2))<0.00001 & abs(edges_3D(:,3)-target_edge(3))<0.00001);
=======
% target_edge = [0.8290, 0.6586, 0.2873];
target_edge = [0.4407, 0.2829, 0.6423];
target_index = find(abs(edges_3D(:,1)-target_edge(1))<0.001 & abs(edges_3D(:,2)-target_edge(2))<0.001 & abs(edges_3D(:,3)-target_edge(3))<0.001);
>>>>>>> e6c6edf843f3f019e8d3420d15a62864487d5a8f
if (length(target_index) > 1)
    error("More than one target index!\n");
end
edge_indices = edge_pair_final(target_index, :);
captured_edge_pairs_list_indx = find(edge_indices ~= -2);
captured_edge_pairs_img_indx = image_indices(captured_edge_pairs_list_indx);
captured_edge_pairs_edg_indices = edge_indices(captured_edge_pairs_list_indx);

edges_locations_metrics = [];
edges_locations_pixels = [];
subplot_cols = 5;
subplot_rows = ceil(length(captured_edge_pairs_img_indx) / subplot_cols);
Rs = [];
Ts = [];
edge_counter = 1;
figure(1);

reprojection_error = zeros(length(captured_edge_pairs_img_indx), 1);
R1_hyp1 = R{captured_edge_pairs_img_indx(1)+1};
T1_hyp1 = T{captured_edge_pairs_img_indx(1)+1};
R1_hyp2 = R{captured_edge_pairs_img_indx(2)+1};
T1_hyp2 = T{captured_edge_pairs_img_indx(2)+1};
edgels_1 = Edgel_files{captured_edge_pairs_img_indx(1)+1};
edges_2D_point_1 = [edgels_1(captured_edge_pairs_edg_indices(1)+1, 1), edgels_1(captured_edge_pairs_edg_indices(1)+1, 2)];
point1 = [edges_2D_point_1(1); edges_2D_point_1(2); 1];
edgels_2 = Edgel_files{captured_edge_pairs_img_indx(2)+1};
edges_2D_point_2 = [edgels_2(captured_edge_pairs_edg_indices(2)+1, 1), edgels_2(captured_edge_pairs_edg_indices(2)+1, 2)];
point2 = [edges_2D_point_2(1); edges_2D_point_2(2); 1];


for i = 1:length(captured_edge_pairs_img_indx)

    % if i == 8 
    %     continue;
    % end

    img_indx = captured_edge_pairs_img_indx(i);     %> start from 0
    edg_indx = captured_edge_pairs_edg_indices(i);  %> start from 0
    img_str = strcat(Image_Path, "/", string(img_indx), "_colors.png");
    img = imread(img_str);
    cols = size(img, 2);

    edgels = Edgel_files{img_indx+1};
    edges_2D_point = [edgels(edg_indx+1, 1), edgels(edg_indx+1, 2)];
    edges_locations_pixels = [edges_locations_pixels, edges_2D_point'];


    Rot = R{img_indx+1};
    Transl = T{img_indx+1};

    Rs = [Rs, reshape(Rot', [1,9])];
    Ts = [Ts; Transl];

    subplot(subplot_rows, subplot_cols, edge_counter);
    imshow(img); hold on;
    fprintf('%d: (%f, %f)\n', img_indx, edges_2D_point(1), edges_2D_point(2));
    plot(edges_2D_point(1), edges_2D_point(2), 'bs'); hold on;

    point_camera = Rot * target_edge' + Transl;
    point_image = K * point_camera;
    edges2D = [point_image(1) / point_image(3), point_image(2) / point_image(3)];
    reprojection_error(i, 1) = sqrt(sum((edges2D - edges_2D_point) .^ 2));

    if(i ~= 1 && i ~= 2)
        %plot epipolar line 1
        Rel_R = Rot * R1_hyp1';
        Rel_T = Transl - Rot * R1_hyp1' * T1_hyp1;
        E21 = skew_T(Rel_T) * Rel_R;
        F21 = K_inv' * E21 * K_inv;
        l = F21 * point1;
        imageWidth = size(img, 2);
        x = [1, imageWidth]; 
        y = (-l(1) * x - l(3)) / l(2); % Y-coordinates
        plot(x, y, 'r', 'LineWidth', 2);hold on;
    
        %plot epipolar line 2
        Rel_R = Rot * R1_hyp2';
        Rel_T = Transl - Rot * R1_hyp2' * T1_hyp2;
        E21 = skew_T(Rel_T) * Rel_R;
        F21 = K_inv' * E21 * K_inv;
<<<<<<< HEAD
        %point1_3 = [point2(1); point2(2); 1];
=======
        point1_3 = [point2(1); point2(2); 1];
>>>>>>> e6c6edf843f3f019e8d3420d15a62864487d5a8f
        l = F21 * point2;
        imageWidth = size(img, 2);
        x = [1, imageWidth]; 
        y = (-l(1) * x - l(3)) / l(2); % Y-coordinates
        plot(x, y, 'b', 'LineWidth', 2);
    end

<<<<<<< HEAD
     if(i == 1)
        %plot epipolar line 2
        Rel_R = Rot * R1_hyp2';
        Rel_T = Transl - Rot * R1_hyp2' * T1_hyp2;
        E21 = skew_T(Rel_T) * Rel_R;
        F21 = K_inv' * E21 * K_inv;
        %point1_3 = [point2(1); point2(2); 1];
        l = F21 * point2;
        imageWidth = size(img, 2);
        x = [1, imageWidth]; 
        y = (-l(1) * x - l(3)) / l(2); % Y-coordinates
        plot(x, y, 'b', 'LineWidth', 2);
    end

    if(i == 2)
        %plot epipolar line 1
        Rel_R = Rot * R1_hyp1';
        Rel_T = Transl - Rot * R1_hyp1' * T1_hyp1;
        E21 = skew_T(Rel_T) * Rel_R;
        F21 = K_inv' * E21 * K_inv;
        l = F21 * point1;
        imageWidth = size(img, 2);
        x = [1, imageWidth]; 
        y = (-l(1) * x - l(3)) / l(2); % Y-coordinates
        plot(x, y, 'r', 'LineWidth', 2);hold on;
    end

=======
>>>>>>> e6c6edf843f3f019e8d3420d15a62864487d5a8f
    edge_counter = edge_counter + 1;

    % if i > 1
    %     R21 = R{i} * R{1}';
    %     T21 = T{i} - R{i}*R{1}'*T{1};
    % 
    %     E21 = skew_T(T21) * R21;
    %     F21 = K_inv' * E21 * K_inv;
    %     a12_img2 = F21(1,:) * edge_point{1}';
    %     b12_img2 = F21(2,:) * edge_point{1}';
    %     c12_img2 = F21(3,:) * edge_point{1}';
    %     yMin12_img2 = -c12_img2./b12_img2;
    %     yMax12_img2 = (-c12_img2 - a12_img2*cols) ./ b12_img2;
    % 
    %     line([1, cols], [yMin12_img2, yMax12_img2], 'Color', 'c', 'LineWidth', 1);
    % end
    pause(0.5);
end

debug = 0;
[corrected_features, reproj_errs, is_sol_global_optimal, Gamma] = ...
    fast_multiview_triangulation_mex(edges_locations_pixels, K_, Rs, Ts, debug);

% figure(3);
% img_str = strcat(Image_Path, "/", string(H1_index), "_colors.png");
% img = imread(img_str);
% edgels = Edgel_files{H1_index+1};
% imshow(img); hold on;
% plot(edgels(:,1), edgels(:,2), 'c.');
