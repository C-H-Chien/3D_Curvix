clear;
close all;

Dataset_Path = "/gpfs/data/bkimia/Datasets/ABC-NEF/";
Object_Name = "00000006/";
Image_Path = fullfile(Dataset_Path, Object_Name, "train_img/");

% Validation view for only image 7 with multiple points
view_index = 7;
orig_points = {[311.5, 491.5], [309.028, 488.973], [310.014, 489.986], [314.463, 494.536]};  % Multiple original points
val_points = {[311.544, 491.545], [311.47, 491.479], [311.397 491.413], [311.272, 491.301]}; % Corresponding validation points
epipolar_lines = {[0.898687, -1, 211.564], [0.898687, -1, 211.564], [0.898687, -1, 211.564], [0.898687, -1, 211.564]}; % Epipolar lines
tangent_lines = {[1.01243, -1, 176.129], [1.02583, -1, 171.963], [1.03184, -1, 170.101 ], [1.01398, -1, 175.677]}; % Tangent lines

% Define colors for different points and lines
point_colors = lines(10); % MATLAB's 'lines' colormap for distinct colors

% Load image 7
image_file = fullfile(Image_Path, sprintf("%d_colors.png", view_index));

if exist(image_file, 'file')
    img = imread(image_file);
    
    figure;
    imshow(img);
    hold on;
    
    % Get image dimensions
    [img_height, img_width, ~] = size(img);
    x_vals = linspace(1, img_width, 100);

    % Loop through all points for image 7
    for j = 1:length(orig_points)
        orig_point = orig_points{j};
        val_point = val_points{j};
        epipolar_line = epipolar_lines{j}; % Epipolar line coefficients [a, b, c]
        tangent_line = tangent_lines{j}; % Tangent line coefficients [a, b, c]

        % Assign unique colors
        color_idx = mod(j, size(point_colors, 1)) + 1;
        point_color = point_colors(color_idx, :);

        % Plot original point
        plot(orig_point(1), orig_point(2), 'o', 'MarkerSize', 8, 'LineWidth', 2, 'Color', point_color);
        
        % Plot validation point
        plot(val_point(1), val_point(2), 'x', 'MarkerSize', 8, 'LineWidth', 2, 'Color', point_color);

        % Plot epipolar line (Dashed)
        if ~isnan(epipolar_line)
            a = epipolar_line(1);
            b = epipolar_line(2);
            c = epipolar_line(3);
            if b ~= 0
                y_vals = (-a * x_vals - c) / b;
                plot(x_vals, y_vals, '--', 'LineWidth', 2, 'Color', point_color);
            end
        end

        % Plot tangent line (Solid)
        if ~isnan(tangent_line)
            a = tangent_line(1);
            b = tangent_line(2);
            c = tangent_line(3);
            if b ~= 0
                y_vals = (-a * x_vals - c) / b;
                plot(x_vals, y_vals, '-', 'LineWidth', 2, 'Color', point_color);
            end
        end
    end
    
    title(sprintf("View %d - Validation vs. Original Points", view_index));

    % Create legend
    legend({'Original Point (o)', 'Corrected Point (x)', 'Epipolar Line', 'Tangent Line'}, ...
        'Location', 'southoutside', 'Orientation', 'horizontal');

    hold off;
else
    fprintf("Image file not found: %s\n", image_file);
end
