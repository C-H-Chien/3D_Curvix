clear;
close all;

Dataset_Path = "/media/chchien/843557f5-9293-49aa-8fb8-c1fc6c72f7ea/home/chchien/datasets/ABC-NEF/";
GT_Folder = "gt_curve_points";
Object_Name = "00000006/";
Image_Path = fullfile(Dataset_Path, Object_Name, "train_img");
Pose_Path = fullfile(Dataset_Path, Object_Name, "RnT");
Edges_Path = fullfile(Dataset_Path, Object_Name, "Edges");
GT_Curve_Points_Path = fullfile(Dataset_Path, GT_Folder, strcat(extractBefore(Object_Name, "/"), ".txt"));
gt_curve_points = importdata(GT_Curve_Points_Path);

%> Some global settings
show_projected_points = 0;
write_refined_camera_poses_to_a_file = 1;

%> Define intrinsic matrix
K = [1111.11136542426,	0,	399.500000000000; 0,	1111.11136542426,	399.500000000000;  0,	0,	1];
refined_Rs = cell(50,1);
refined_Ts = cell(50,1);

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

%> Flip the entire 3D curve points due to the conversion of the rotation
%  matrix from "improper" to "proper"
flipped_gt_curve_points = diag([1,1,-1])*gt_curve_points';

for img_index = 0:49
    fprintf(strcat("image ", string(img_index), ": "));
    edgels = Edgel_files{img_index+1};

    %> Make the rotation matrix "proper"
    R{img_index+1} = R{img_index+1} * diag([1,1,-1]);
    
    R_ = R{img_index+1};
    T_ = T{img_index+1};
    
    original_proj_pts_on_H1 = K * (R_ * flipped_gt_curve_points + T_);
    original_proj_pts_on_H1 = original_proj_pts_on_H1 ./ original_proj_pts_on_H1(3,:);
    
    %> Refine through iteratively defining the correspondences through nearest
    %  neighbor
    fprintf("iter ");
    for iter = 1:5
        fprintf(strcat(string(iter), ".."));
    
        proj_pts_on_H1 = K * (R_ * flipped_gt_curve_points + T_);
        proj_pts_on_H1 = proj_pts_on_H1 ./ proj_pts_on_H1(3,:);
    
        index_pair_3D_2D = [];
        for i = 1:size(edgels, 1)
            [min_val, min_index] = min(vecnorm(proj_pts_on_H1(1:2,:) - edgels(i,1:2)', 2, 1));
            if min_val <= 2
                index_pair_3D_2D = [index_pair_3D_2D; [min_index, i]];
            end
        end
        
        paired_GT_3D_points = flipped_gt_curve_points(:,index_pair_3D_2D(:,1));
        paired_2D_edge_points = edgels(index_pair_3D_2D(:,2), :)';
        
        x0 = zeros(1, 6);
        eul = rotm2eul(R{img_index+1});
        x0(1:3) = eul;
        x0(4:6) = T{img_index+1}';
        
        ff = @(x)(getProjectionError(paired_GT_3D_points, paired_2D_edge_points, K, x));
        options = optimoptions('lsqnonlin', 'Display','off');
        options.Algorithm = 'levenberg-marquardt';
        pose_output = lsqnonlin(ff, x0, [], [], options);
        
        refined_R = eul2rotm(pose_output(1:3));
        refined_T = pose_output(4:6);
        
        R_ = refined_R;
        T_ = refined_T';
    end
    fprintf("\n");

    %> store the optimized camera poses
    refined_Rs{img_index+1} = refined_R;
    refined_Ts{img_index+1} = refined_T;

    if show_projected_points == 1
        new_proj_pts_on_H1 = K * (refined_R * flipped_gt_curve_points + refined_T');
        new_proj_pts_on_H1 = new_proj_pts_on_H1 ./ new_proj_pts_on_H1(3,:);
        
        figure(img_index+1);
        img_H1_str = strcat(Image_Path, "/", string(img_index), "_colors.png");
        img_H1 = imread(img_H1_str);
        imshow(img_H1); hold on;
        plot(edgels(:,1), edgels(:,2), 'c.'); hold on;
        plot(original_proj_pts_on_H1(1,:), original_proj_pts_on_H1(2,:), 'g.'); hold on;
        plot(new_proj_pts_on_H1(1,:), new_proj_pts_on_H1(2,:), 'm.');
        % plot(edgels(index_pair_3D_2D(:,2),1), edgels(index_pair_3D_2D(:,2),2), 'co'); hold on;
        % plot(proj_pts_on_H1(1,index_pair_3D_2D(:,1)), proj_pts_on_H1(2,index_pair_3D_2D(:,1)), 'go');
        hold off;
    end
end



if write_refined_camera_poses_to_a_file == 1
    file_refined_R = fopen('/home/chchien/BrownU/research/EdgeSketchGrouping/refine_camera_poses/refined_R.txt', 'w');
    file_refined_T = fopen('/home/chchien/BrownU/research/EdgeSketchGrouping/refine_camera_poses/refined_T.txt', 'w');
    for i = 1:50
        wR = refined_Rs{i};
        wT = refined_Ts{i};
        for m = 1:3
            for n = 1:3
                fprintf(file_refined_R, '%6s\t', num2str(wR(m,n), '%.10f'));
            end
            fprintf(file_refined_R, '\n');
        end
        for m = 1:3
            fprintf(file_refined_T, '%6s\n', num2str(wT(m), '%.10f'));
        end
    end
    fclose(file_refined_R);
    fclose(file_refined_T);
end

% yy = load("gt_curves_points.mat");
% figure(1);
% plot3(gt_curve_points(:,1), gt_curve_points(:,2), gt_curve_points(:,3), 'g.');
% axis equal;
% xlabel("x");
% ylabel("y");
% zlabel("z");
% set(gcf, 'color', 'w');

% figure(2);
% plot3(gt_curve_points(:,1), gt_curve_points(:,2), gt_curve_points(:,3), 'g.');
% axis equal;
% xlabel("x");
% ylabel("y");
% zlabel("z");
% set(gcf, 'color', 'w');