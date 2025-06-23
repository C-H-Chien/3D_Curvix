clear;
close all;

image_num = 28;
% Define dataset and image paths
Dataset_Path = "/gpfs/data/bkimia/Datasets/ABC-NEF/";
Object_Name = "00000006/";
Image_Path = fullfile(Dataset_Path, Object_Name, "train_img");
Edges_Path = fullfile(Dataset_Path, Object_Name, "Edges");
Edgel_file = importdata(strcat(Edges_Path, "/Edge_", string(image_num), "_t1.txt"));
Output_Path = fullfile(fileparts(mfilename('fullpath')), '..', "outputs/");
H1_edges = importdata(strcat(Output_Path, "test_epipolar_line_from_H1_edges.txt"));
H2_edges = importdata(strcat(Output_Path, "test_H2_edges_status.txt"));

%> extract data
epipolar_line_coeffs = H1_edges(:,4:6);
H2_edges_group_indices = H2_edges(:,1);

img_path = sprintf('/gpfs/data/bkimia/Datasets/ABC-NEF/00000006/train_img/%d_colors.png',  image_num);

mag = 0.5;
for i = 1:size(H1_edges,1)
    figure;
    img = imread(img_path);
    imshow(img); hold on;

    extract_indx = find(H2_edges_group_indices == i-1);
    extracted_H2_edges = H2_edges(extract_indx, :); 
    correction_status = H2_edges(extract_indx, end);
    target_indx = find(correction_status == 2);

    %> plot epipolar line
    l = H1_edges(i,4:6);
    imageWidth = size(img, 2);
    x = [1, imageWidth]; 
    y = (-l(1) * x - l(3)) / l(2); % Y-coordinates
    plot(x, y, 'y', 'LineWidth', 2);hold on;

    original_H2_edges = extracted_H2_edges(target_indx,2:4);
    corrected_H2_edges = extracted_H2_edges(target_indx,5:7);

    original_theta = original_H2_edges(:,3);
    original_u = cos(original_theta) * mag;
    original_v = sin(original_theta) * mag;

    corrected_theta = corrected_H2_edges(:,3);
    corrected_u = cos(corrected_theta) * mag;
    corrected_v = sin(corrected_theta) * mag;
    
    %> Plot edges as arrows
    quiver(original_H2_edges(:,1), original_H2_edges(:,2), original_u, original_v, 0, 'c-', 'LineWidth', 1.2); 
    hold on;
    quiver(corrected_H2_edges(:,1), corrected_H2_edges(:,2), corrected_u, corrected_v, 0, 'm-', 'LineWidth', 1.2); 
    hold on;
    plot(original_H2_edges(:,1), original_H2_edges(:,2), 'co', 'LineWidth', 1.2); hold on;
    plot(corrected_H2_edges(:,1), corrected_H2_edges(:,2), 'mo', 'LineWidth', 1.2); hold on;

    quiver(original_H2_edges(:,1), original_H2_edges(:,2), corrected_u, corrected_v, 0, 'g-', 'LineWidth', 1.2); 
    hold off;

    % waitforbuttonpress;
    % close all;
end



