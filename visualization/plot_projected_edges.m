clear;
close all;

%> Define the folder path containing the 3D edge and tangent output files
data_folder_name = 'outputs';
data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);

%> Edge view folder for observed edge points
observed_edge_folder = '/gpfs/data/bkimia/zqiwu/3D/3D_Edge_Sketch_and_Grouping/datasets/ABC-NEF/00000006/Edges/';

views = [7];

%> Bounding box range
x_min = 312; x_max = 320;
y_min = 480; y_max = 500;
mag = 0.1;

% x_min = 290; x_max = 300;
% y_min = 390; y_max = 420;
% mag = 0.6;


for i = 1:length(views)
    view_id = views(i);
    
    %> Construct the file path for the corresponding image
    img_path = sprintf('/gpfs/data/bkimia/Datasets/ABC-NEF/00000006/train_img/%d_colors.png', view_id);
    img = imread(img_path);

    figure;
    imshow(img); hold on;
    title(sprintf('Projected and Observed Edges on View %d', view_id));

    %> Load and plot observed 2D edge points (1st and 2nd columns only)
    observed_file = sprintf('Edge_%d_t1.txt', view_id);
    observed_path = fullfile(observed_edge_folder, observed_file);
    if isfile(observed_path)
        observed_edges = dlmread(observed_path);
        in_bbox = observed_edges(:,1) >= x_min-5 & observed_edges(:,1) <= x_max+5 & ...
          observed_edges(:,2) >= y_min-5 & observed_edges(:,2) <= y_max+5;

        if any(in_bbox)
            points = observed_edges(in_bbox, 1:2);
            tangents = observed_edges(in_bbox, 3);  % Orientation (assumed in radians)
        
            % Plot observed points
            plot(points(:,1), points(:,2), 'go', 'MarkerSize', 8, 'DisplayName', 'Observed Edges');
        
            % Plot green orientation lines
            for i = 1:size(points, 1)
                x = points(i, 1);
                y = points(i, 2);
                theta = tangents(i);  % radians
        
                % Draw orientation line in green
                plot([x + mag*cos(theta), x - mag*cos(theta)], [y + mag*sin(theta), y - mag*sin(theta)],'g-', 'LineWidth', 1.2);
            end
        end

        fprintf('Observed points in bounding box for view %d: %d\n', view_id, sum(in_bbox));

        % if observed_edges(:,1) >= x_min & observed_edges(:,1) <= x_max & observed_edges(:,2) >= y_min & observed_edges(:,2) <= y_max
        %     plot(observed_edges(:,1), observed_edges(:,2), 'g.', 'MarkerSize', 8, 'DisplayName', 'Observed Edges');
        % end
    else
        warning('Observed edge file not found: %s', observed_path);
    end

    %> Load and plot projected 3D edges from the graph
    %projected_path = fullfile(data_folder_path, '3D_edge_groups_projected.txt');
    projected_path = fullfile(data_folder_path, '3D_edge_groups_projected.txt');
    if ~isfile(projected_path)
        warning('Projected edge file not found: %s', projected_path);
        continue;
    end

    projected_edges = dlmread(projected_path);  % Format: u1 v1 u2 v2 group_id

    for j = 1:size(projected_edges, 1)
        u1 = projected_edges(j, 1);
        v1 = projected_edges(j, 2);
        u2 = projected_edges(j, 3);
        v2 = projected_edges(j, 4);
        t1x = projected_edges(j, 5);
        t1y = projected_edges(j, 6);
        t2x = projected_edges(j, 7);
        t2y = projected_edges(j, 8);
        label = projected_edges(j, 9); 
        weight = projected_edges(j, 10); 



        if u1 >= x_min & u1 <= x_max & v1 >= y_min & v1 <= y_max  
            % Plot endpoints in blue
            plot([u1, u2], [v1, v2], 'b.', 'MarkerSize', 10, 'MarkerFaceColor', 'b');

            plot([u1 + mag*t1x, u1 - mag*t1x], [v1 + mag*t1y, v1 - mag*t1y], 'c-', 'LineWidth', 1.2);
            plot([u2 + mag*t2x, u2 - mag*t2x], [v2 + mag*t2y, v2 - mag*t2y], 'c-', 'LineWidth', 1.2);

            % % Plot tangent centered at (u1, v1)
            % plot([u1 + mag*t1x, u1 - mag*t1x], [v1 + mag*t1y, v1 - mag*t1y],'c-', 'LineWidth', 1.2);  % cyan line for clarity
            % 
            % % Plot tangent centered at (u2, v2)
            % plot([u2 + mag*t2x, u2 - mag*t2x], [v2 + mag*t2y, v2 - mag*t2y], 'c-', 'LineWidth', 1.2);

            if label == 1
                plot([u1, u2], [v1, v2], 'Color', [0.5 0 0.5], 'LineWidth', 1.5); % purple
            elseif label == 0
                plot([u1, u2], [v1, v2], 'r-', 'LineWidth', 1.5); % red
            else
                plot([u1, u2], [v1, v2], 'k-', 'LineWidth', 1); % default: black for in-between cases
            end
        end
    end

    %legend('Observed Edges', '3D Projected Edges');
    fprintf('Plotted projected edges for view %d from %s\n', view_id, projected_path);
    pause(0.5);
end
