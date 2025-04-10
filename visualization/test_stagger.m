clear;
close all;

% Define dataset and image paths
Dataset_Path = "/gpfs/data/bkimia/Datasets/ABC-NEF/";
Object_Name = "00000006/";
Image_Path = fullfile(Dataset_Path, Object_Name, "train_img");

% Define the validation views and points for both groups
validation_views_points_group_1 = {
    39, [342.542, 511.939], [342.542, 511.939], [342.542, 511.939], [342.542, 511.939];
    7,  [311.5, 491.5], [309.028, 488.973], [310.014, 489.986], [314.463, 494.536];
    21, [493.941, 480.478], [493.554, 481.519], [493.554, 481.519], [493.941, 480.478];
    24, [356.474, 431.791], [355.981, 431.859], [355.981, 431.859], [356.474, 431.791];
    36, [375.493, 426.91], [375.989, 426.864], [375.493, 426.91], [375.493, 426.91];
    49, [514.411, 422.533], [514.842, 423.567], [514.842, 423.567], [514.411, 422.533];
};

validation_views_corrected_points_group_1 = {
    39, [342.542, 511.939], [342.542, 511.939], [342.542, 511.939], [342.542, 511.939];
    7,  [311.544, 491.545], [311.47, 491.479], [311.397, 491.413], [311.272, 491.301];
    21, [493.804, 480.838], [493.794, 480.822], [493.794, 480.822], [493.804, 480.838];
    24, [356.196, 431.825], [356.196, 431.828], [356.196, 431.828], [356.196, 431.825];
    36, [376.138, 426.851], [376.139, 426.849], [376.138, 426.851], [376.138, 426.851];
    49, [514.515, 422.814], [514.499, 422.818], [514.499, 422.818], [514.515, 422.814];
};

validation_views_points_group_2 = {
    39, [342.899, 512.148], [342.899, 512.148], [342.899, 512.148], [342.899, 512.148];
    7,  [307.798, 487.7], [308.294, 488.203], [308.788, 488.708], [312.49, 492.509];
    21, [493.779, 480.92], [493.705, 481.073], [493.554, 481.519], [493.554, 481.519];
    24, [356.474, 431.791], [355.981, 431.859], [355.981, 431.859], [356.474, 431.791];
    36, [375.989, 426.864], [376.482, 426.813], [375.989, 426.864], [375.989, 426.864];
    49, [514.411, 422.533], [514.565, 422.975], [514.842, 423.567], [514.842, 423.567];
};

validation_views_corrected_points_group_2 = {
    39, [342.899, 512.148], [342.899, 512.148], [342.899, 512.148], [342.899, 512.148];
    7,  [311.846, 491.805], [311.751, 491.719], [311.593, 491.577], [311.545, 491.534];
    21, [493.67, 481.219], [493.659, 481.202], [493.661, 481.206], [493.661, 481.206];
    24, [356.196, 431.825], [356.196, 431.828], [356.196, 431.828], [356.196, 431.825];
    36, [376.532, 426.81], [376.532, 426.808], [376.532, 426.81], [376.532, 426.81];
    49, [514.62, 423.096], [514.614, 423.098], [514.626, 423.095], [514.626, 423.095];
};

% Number of validation images
num_views = size(validation_views_points_group_1, 1);
num_cols = ceil(sqrt(num_views)); 
num_rows = ceil(num_views / num_cols);

figure;
for i = 1:num_views
    % Extract the validation view index
    view_idx = validation_views_points_group_1{i, 1};
    
    % Extract original and corrected points from both groups
    original_pts_1 = validation_views_points_group_1(i, 2:end);
    corrected_pts_1 = validation_views_corrected_points_group_1(i, 2:end);
    original_pts_2 = validation_views_points_group_2(i, 2:end);
    corrected_pts_2 = validation_views_corrected_points_group_2(i, 2:end);

    % Load the corresponding image
    img_str = fullfile(Image_Path, sprintf("%d_colors.png", view_idx));
    img = imread(img_str);

    % Display image
    subplot(num_rows, num_cols, i);
    imshow(img);
    hold on;

    % Iterate over all 4 edges in this image for Group 1 (Red)
    for j = 1:size(original_pts_1, 2)
        plot(original_pts_1{j}(1), original_pts_1{j}(2), 'rx', 'MarkerSize', 8, 'LineWidth', 2);
        plot(corrected_pts_1{j}(1), corrected_pts_1{j}(2), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    end

    % Iterate over all 4 edges in this image for Group 2 (Blue)
    for j = 1:size(original_pts_2, 2)
        plot(original_pts_2{j}(1), original_pts_2{j}(2), 'bx', 'MarkerSize', 8, 'LineWidth', 2);
        plot(corrected_pts_2{j}(1), corrected_pts_2{j}(2), 'bo', 'MarkerSize', 8, 'LineWidth', 2);
    end

    % Add title with index
    title(sprintf("View %d", view_idx), 'FontSize', 10);
    hold off;
end
