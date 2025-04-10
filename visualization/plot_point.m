clear;
close all;

% Define dataset and image paths
Dataset_Path = "/gpfs/data/bkimia/Datasets/ABC-NEF/";
Object_Name = "00000006/";
Image_Path = fullfile(Dataset_Path, Object_Name, "train_img");
Pose_Path = fullfile(Dataset_Path, Object_Name, "RnT");

%> Parsing object data
R_file = importdata(fullfile(Pose_Path, "R_matrix.txt"));
T_file = importdata(fullfile(Pose_Path, "T_matrix.txt"));

R = cell(50,1);
T = cell(50,1);
for ei = 0:49
    R{ei+1} = R_file(3*ei+1:3*ei+3, :);
    T{ei+1} = T_file(3*ei+1:3*ei+3);
end

% Camera intrinsic matrix
K = [1111.11136542426, 0, 399.500000000000;
     0, 1111.11136542426, 399.500000000000;
     0, 0, 1];
K_inv = inv(K);

% Define image indices and corresponding points
image_data = {
    37, [519.863, 399.004, 1, 1.53603], [519.781, 398.748, 1, 1.53603];
    47, [517.958, 384.825, 1, 1.47499], [518.206, 384.938, 1, 1.47499];
    6,  [244.691, 347.527, 1, -1.43560], [246.096, 347.619, 1, -1.43560];
    20, [528.107, 393.502, 1, 1.593804], [528.301, 393.579, 1, 1.593804];
    37, [516.334, 387.002, 1, 1.557710], [516.657, 386.403, 1, 1.557710];
    43, [511.968, 363.005, 1, 1.397898], [512.4, 363.013, 1, 1.397898];
    46, [513.349, 416.495, 1, 1.60077], [513.821, 416.416, 1, 1.60077];
    49, [517.213, 411.471, 1, 1.429097], [517.255, 412.216, 1, 1.429097];
};

num_images = size(image_data, 1);
num_cols = ceil(sqrt(num_images)); 
num_rows = ceil(num_images / num_cols);

R1 = R{image_data{1, 1}+1};
T1 = T{image_data{1, 1}+1};
R2 = R{image_data{2, 1}+1};
T2 = T{image_data{2, 1}+1};

R21 = R2 * R1';
T21 = T2 - R2 * R1' * T1;
orig_hy1 = image_data{1, 2}(1:3);
orig_hy2 = image_data{2, 2}(1:3);
corr_hy1 = image_data{1, 3}(1:3);
corr_hy2 = image_data{2, 3}(1:3);
tangent_hyp1 = image_data{1, 2}(4);
tangent_hyp2 = image_data{2, 2}(4);
mag = 0.25;

figure;
for i = 1:num_images
    image_idx = image_data{i, 1};
    orig_pt = image_data{i, 2}';
    corr_pt = image_data{i, 3}';
    tangent = orig_pt(4);
    
    img_str = fullfile(Image_Path, sprintf("%d_colors.png", image_idx));
    
    subplot(num_rows, num_cols, i);
    img = imread(img_str);
    imshow(img);
    hold on;

    plot(orig_pt(1), orig_pt(2), 'ro', 'MarkerSize', 8, 'LineWidth', 2); hold on;
    plot(corr_pt(1), corr_pt(2), 'rx', 'MarkerSize', 8, 'LineWidth', 2); hold on;

    plot([orig_pt(1) + mag * cos(tangent), orig_pt(1) - mag * cos(tangent)], ...
         [orig_pt(2) + mag * sin(tangent), orig_pt(2) - mag * sin(tangent)], ...
         'r', 'LineWidth', 2); hold on;
    plot([corr_pt(1) + mag * cos(tangent), corr_pt(1) - mag * cos(tangent)], ...
         [corr_pt(2) + mag * sin(tangent), corr_pt(2) - mag * sin(tangent)], ...
         'r', 'LineWidth', 2); hold on;

    if(i ~= 1 && i ~= 2)
        R3 = R{image_idx+1};
        T3 = T{image_idx+1};
        R31 = R3 * R1';
        T31 = T3 - R3 * R1'* T1;
    
        [t_orig, reprojected_orig_point] = compute_reprojection(K_inv*orig_hy1', K_inv*orig_hy2', tangent_hyp1, tangent_hyp2, R21, T21, R31, T31);
        [t_corr, reprojected_corr_point] = compute_reprojection(K_inv*corr_hy1', K_inv*corr_hy2', tangent_hyp1, tangent_hyp2, R21, T21, R31, T31);
        reprojected_orig_point = K*reprojected_orig_point ;
        reprojected_corr_point = K*reprojected_corr_point ;
        plot(reprojected_orig_point(1)/reprojected_orig_point(3), reprojected_orig_point(2)/reprojected_orig_point(3), 'bo', 'MarkerSize', 8, 'LineWidth', 2);
        plot(reprojected_corr_point(1)/reprojected_corr_point(3), reprojected_corr_point(2)/reprojected_corr_point(3), 'bx', 'MarkerSize', 8, 'LineWidth', 2);

        plot([reprojected_orig_point(1) + mag * t_orig(1), reprojected_orig_point(1) - mag * t_orig(1)], ...
         [reprojected_orig_point(2) + mag * t_orig(2), reprojected_orig_point(2) - mag * t_orig(2)], ...
         'b', 'LineWidth', 2); hold on;
        plot([reprojected_corr_point(1) + mag * t_corr(1), reprojected_corr_point(1) - mag * t_corr(1)], ...
         [reprojected_corr_point(2) + mag * t_corr(2), reprojected_corr_point(2) - mag * t_corr(2)], ...
         'b', 'LineWidth', 2); hold on;

    end
    % Add title
    title(sprintf("Image %d", image_idx), 'FontSize', 10);
    hold off;
end



function [t3, gamma3] = compute_reprojection(gamma1, gamma2, theta1, theta2, R21, T21, R31, T31)
    b3 = [0; 0; 1];  
    t1 = [cos(theta1); sin(theta1); 0];
    t2 = [cos(theta2); sin(theta2); 0];

    num_gamma3 = ((b3'*T21) * (b3'*R21'*gamma2) - (b3'*R21'*T21)) * (R31*gamma1) + (1 - ((b3'*R21*gamma1) * (b3'*R21'*gamma2))) * T31;
    deno_gamma3 = ((b3'*T21) * (b3'*R21'*gamma2) - (b3'*R21'*T21)) * (b3' * R31 * gamma1) + (1 - ((b3' * R21 * gamma1) * (b3' * R21' * gamma2))) * (b3' * T31);
    gamma3 = num_gamma3 / deno_gamma3;  
    
    cross_term_1 = cross(gamma1, t1);
    cross_term_2 = cross(t2, gamma2);
    num_t3 = R31 * cross(cross_term_1, R21'*cross_term_2) - b3' * cross(cross_term_1, R21'*cross_term_2)*gamma3;
    deno_t3 = norm(num_t3);
    t3 = num_t3 / deno_t3;
end

