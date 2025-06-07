%% Visualize 3D Curves from Connectivity Graph with Curve Joining
% This script visualizes the curves extracted from the connectivity graph
% and joins curves whose endpoints are close to each other

% Define the folder path
data_folder_name = 'outputs';
data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);
curve_file = fullfile(data_folder_path, 'curves_from_connectivity_graph.txt');
%curve_file = fullfile(data_folder_path, 'aligned_orientation.txt');

% Read the data
fileID = fopen(curve_file, 'r');
if fileID == -1
    error('Could not open file: %s', curve_file);
end

% Skip the header lines
header = fgetl(fileID);
header2 = fgetl(fileID);

% Initialize data structures
curve_data = [];
curve_ids = [];
node_indices = [];
current_curve_id = -1;
mag = 0; % Magnitude for orientation vectors
% mag = 0.0005;

% Read line by line
line = fgetl(fileID);
while ischar(line)
    % Skip empty lines
    if ~isempty(line)
        data = sscanf(line, '%f');
        if length(data) >= 8
            curve_id = data(1);
            node_idx = data(2);
            x = data(3);
            y = data(4);
            z = data(5);
            dir_x = data(6);
            dir_y = data(7);
            dir_z = data(8);

            curve_data = [curve_data; x, y, z, dir_x, dir_y, dir_z];
            curve_ids = [curve_ids; curve_id];
            node_indices = [node_indices; node_idx];            

            
            if curve_id ~= current_curve_id
                current_curve_id = curve_id;
            end
        end
    end
    line = fgetl(fileID);
end
fclose(fileID);

% Get unique curve IDs
unique_curves = unique(curve_ids);
num_curves = length(unique_curves);
fprintf('Found %d original curves\n', num_curves);

%% Create a structure to store curve points and directions
curves = struct('points', cell(num_curves, 1), 'directions', cell(num_curves, 1), 'id', cell(num_curves, 1));

% Populate the curves structure
for i = 1:num_curves
    curve_idx = curve_ids == unique_curves(i);
    curves(i).points = curve_data(curve_idx, 1:3);
    curves(i).directions = curve_data(curve_idx, 4:6);
    curves(i).id = unique_curves(i);
end

% Print the bounding box of all curves for debugging
all_points = vertcat(curves(:).points);
min_xyz = min(all_points, [], 1);
max_xyz = max(all_points, [], 1);


distance_threshold = 0.002;
min_curve_size = 5;

%% Find curves that should be joined
% Create a bidirectional graph to represent connections between curves
connection_graph = zeros(num_curves, num_curves);
connection_types = cell(num_curves, num_curves); % Store the type of connection

% Check for all possible endpoint connections
for i = 1:num_curves
    curve_i_start = curves(i).points(1, :);
    curve_i_end = curves(i).points(end, :);
    
    for j = 1:num_curves
        if i ~= j
            curve_j_start = curves(j).points(1, :);
            curve_j_end = curves(j).points(end, :);
            
            % Calculate distances between all possible endpoint combinations
            distances = [
                norm(curve_i_end - curve_j_start),    % end-to-start 
                norm(curve_i_start - curve_j_end),    % start-to-end
                norm(curve_i_start - curve_j_start),  % start-to-start
                norm(curve_i_end - curve_j_end)       % end-to-end
            ];
            
            connection_types_options = {'end-start', 'start-end', 'start-start', 'end-end'};
            
            [min_distance, min_idx] = min(distances);
            
            % If any distance is below threshold, mark these curves to be joined
            if min_distance < distance_threshold
                connection_graph(i, j) = 1;
                connection_types{i, j} = connection_types_options{min_idx};
                % fprintf('Joining curve %d to curve %d (%s connection, distance: %.6f)\n', ...
                %     curves(i).id, curves(j).id, connection_types{i, j}, min_distance);
            end
        end
    end
end

% Identify groups of curves that should be joined
joined_curves = {};
visited = false(1, num_curves);

% Find all connected components
for i = 1:num_curves
    if ~visited(i)
        % Start DFS from this node
        component = [];
        stack = i;
        
        while ~isempty(stack)
            node = stack(end);
            stack(end) = [];
            
            if ~visited(node)
                visited(node) = true;
                component = [component, node];
                
                % Add all unvisited neighbors to the stack
                for neighbor = 1:num_curves
                    if (connection_graph(node, neighbor) || connection_graph(neighbor, node)) && ~visited(neighbor)
                        stack = [stack, neighbor];
                    end
                end
            end
        end
        
        if ~isempty(component)
            joined_curves{end+1} = component;
        end
    end
end

fprintf('After joining, found %d curve groups\n', length(joined_curves));


figure();
hold on;
fixed_colors = jet(30);
total_points = 0;
curves_plotted = 0;
curves_skipped = 0;

% Plot each joined curve group
for i = 1:length(joined_curves)
    group = joined_curves{i};
    
    % If it's a single curve with no joins
    if length(group) == 1
        curve_idx = group;
        points = curves(curve_idx).points;
        directions = curves(curve_idx).directions;
        
        % Skip if curve is too small
        if size(points, 1) < min_curve_size
            fprintf('Skipping single curve %d with only %d points (minimum is %d)\n', ...
                curves(curve_idx).id, size(points, 1), min_curve_size);
            curves_skipped = curves_skipped + 1;
            continue;
        end
    else
        % Create a working copy of the curves for this group
        group_curves = struct('points', cell(length(group), 1), 'directions', cell(length(group), 1), 'id', cell(length(group), 1));
        for j = 1:length(group)
            group_curves(j) = curves(group(j));
        end
        
        % Start with the first curve
        final_curves = [];
        final_directions = [];
        used = false(1, length(group));
        
        % Keep finding and connecting curves until all are used
        while sum(~used) > 0
            % If we're starting a new chain, pick the first unused curve
            if isempty(final_curves)
                first_unused = find(~used, 1);
                final_curves = group_curves(first_unused).points;
                final_directions = group_curves(first_unused).directions;
                used(first_unused) = true;
                fprintf('Starting new chain with curve %d (group %d)\n', group_curves(first_unused).id, i);
            end
            
            % Try to find the next curve to connect
            best_distance = inf;
            best_curve = -1;
            best_connection = '';
            current_start = final_curves(1, :);
            current_end = final_curves(end, :);
            
            for j = 1:length(group)
                if ~used(j)
                    next_start = group_curves(j).points(1, :);
                    next_end = group_curves(j).points(end, :);
                    
                    % Check all possible connections
                    distances = [
                        norm(current_end - next_start),    % current end to next start
                        norm(current_start - next_end),    % current start to next end
                        norm(current_start - next_start),  % current start to next start
                        norm(current_end - next_end)       % current end to next end
                    ];
                    
                    connection_options = {'end-start', 'start-end', 'start-start', 'end-end'};
                    
                    [min_dist, min_idx] = min(distances);
                    
                    if min_dist < best_distance && min_dist < distance_threshold
                        best_distance = min_dist;
                        best_curve = j;
                        best_connection = connection_options{min_idx};
                    end
                end
            end
            
            % If we found a good connection, add the curve appropriately
            if best_curve > 0
                next_points = group_curves(best_curve).points;
                next_directions = group_curves(best_curve).directions;
                used(best_curve) = true;
                
                % Store current size for debugging
                prev_size = size(final_curves, 1);
                
                switch best_connection
                    case 'end-start'
                        % Add to end (normal case)
                        final_curves = [final_curves; next_points];
                        final_directions = [final_directions; next_directions];
                    case 'start-end'
                        % Add to start (reverse current)
                        final_curves = [flipud(next_points); final_curves];
                        final_directions = [flipud(-next_directions); final_directions];
                    case 'start-start'
                        % Add to start (reverse next)
                        final_curves = [flipud(next_points); final_curves];
                        final_directions = [flipud(-next_directions); final_directions];
                    case 'end-end'
                        % Add to end (reverse next)
                        final_curves = [final_curves; flipud(next_points)];
                        final_directions = [final_directions; flipud(-next_directions)];
                end
                
                % Debug output
                fprintf('Added curve %d to chain with %s connection (distance: %.6f, added %d points)\n', ...
                    group_curves(best_curve).id, best_connection, best_distance, size(final_curves, 1) - prev_size);
            else
                % If no good connection found, save this chain
                if ~isempty(final_curves)
                    % Skip if curve is too small
                    if size(final_curves, 1) < min_curve_size
                        fprintf('Skipping chain in group %d with only %d points (minimum is %d)\n', ...
                            i, size(final_curves, 1), min_curve_size);
                        curves_skipped = curves_skipped + 1;
                    else
                        % Plot the curve
                        color_idx = mod(i-1, 30) + 1;
                        color = fixed_colors(color_idx, :);
                        
                        % Add to total point count
                        total_points = total_points + size(final_curves, 1);
                        curves_plotted = curves_plotted + 1;
                        
                        % Print curve bounds for debugging
                        min_pts = min(final_curves, [], 1);
                        max_pts = max(final_curves, [], 1);
                        fprintf('Plotting curve group %d with %d points - X: %.4f to %.4f, Y: %.4f to %.4f, Z: %.4f to %.4f\n', ...
                            i, size(final_curves, 1), min_pts(1), max_pts(1), min_pts(2), max_pts(2), min_pts(3), max_pts(3));
                        
                        % Plot
                        h = plot3(final_curves(:, 1), final_curves(:, 2), final_curves(:, 3), '-', 'LineWidth', 2, 'Color', color);
                        if isempty(h) || ~isvalid(h)
                            warning('Failed to plot curve group %d', i);
                        end
                        
                        % Also plot direction vectors if specified
                        if mag > 0
                            quiver3(final_curves(:,1), final_curves(:,2), final_curves(:,3), ...
                                mag*final_directions(:,1), mag*final_directions(:,2), mag*final_directions(:,3), ...
                                0, 'Color', 'k', 'LineWidth', 0.5, 'MaxHeadSize', 0.5);
                        end
                    end
                    
                    % Reset for next chain in same group (if any)
                    final_curves = [];
                    final_directions = [];
                end
            end
        end
        
        % Plot the last chain if it exists
        if ~isempty(final_curves)
            % Skip if curve is too small
            if size(final_curves, 1) < min_curve_size
                fprintf('Skipping final chain in group %d with only %d points (minimum is %d)\n', ...
                    i, size(final_curves, 1), min_curve_size);
                curves_skipped = curves_skipped + 1;
            else
                color_idx = mod(i-1, 30) + 1;
                color = fixed_colors(color_idx, :);
                
                % Add to total point count
                total_points = total_points + size(final_curves, 1);
                curves_plotted = curves_plotted + 1;
                
                % Print curve bounds for debugging
                min_pts = min(final_curves, [], 1);
                max_pts = max(final_curves, [], 1);
                fprintf('Plotting final curve in group %d with %d points - X: %.4f to %.4f, Y: %.4f to %.4f, Z: %.4f to %.4f\n', ...
                    i, size(final_curves, 1), min_pts(1), max_pts(1), min_pts(2), max_pts(2), min_pts(3), max_pts(3));
                
                % Plot
                h = plot3(final_curves(:, 1), final_curves(:, 2), final_curves(:, 3), '-', 'LineWidth', 2, 'Color', color);
                if isempty(h) || ~isvalid(h)
                    warning('Failed to plot final curve in group %d', i);
                end
                
                % Also plot direction vectors if specified
                if mag > 0
                    quiver3(final_curves(:,1), final_curves(:,2), final_curves(:,3), ...
                        mag*final_directions(:,1), mag*final_directions(:,2), mag*final_directions(:,3), ...
                        0, 'Color', 'k', 'LineWidth', 0.5, 'MaxHeadSize', 0.5);
                end
            end
        end
    end
    
    % If we're dealing with a single curve (no joins needed)
    if length(group) == 1 && ~isempty(points) && size(points, 1) >= min_curve_size
        color_idx = mod(i-1, 30) + 1;
        color = fixed_colors(color_idx, :);
        
        % Add to total point count
        total_points = total_points + size(points, 1);
        curves_plotted = curves_plotted + 1;
        
        % Print curve bounds for debugging
        min_pts = min(points, [], 1);
        max_pts = max(points, [], 1);
        % Plot
        h = plot3(points(:, 1), points(:, 2), points(:, 3), '-', 'LineWidth', 2, 'Color', color);
        if isempty(h) || ~isvalid(h)
            warning('Failed to plot single curve %d', curves(group).id);
        end
        
        % Also plot direction vectors if specified
        if mag > 0
            quiver3(points(:,1), points(:,2), points(:,3), ...
                mag*directions(:,1), mag*directions(:,2), mag*directions(:,3), ...
                0, 'Color', 'k', 'LineWidth', 0.5, 'MaxHeadSize', 0.5);
        end
    end
end

fprintf('Total curves plotted: %d\n', curves_plotted);

title('3D Curves from Connectivity Graph with Joined Curves');
view(3); 
set(gcf, 'color', 'w');
ax = gca;
ax.Clipping = "off";
axis off;
axis equal;

% Make sure everything is visible
drawnow;