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

%% Define the distance threshold for joining curves
distance_threshold = 0.015; 

% Define minimum curve size for plotting
min_curve_size = 5;
fprintf('Only plotting curves with %d or more points\n', min_curve_size);

%% Find curves that should be joined
% Create a directed graph to represent connections between curves
connection_graph = zeros(num_curves, num_curves);

% Check for close endpoints
for i = 1:num_curves
    curve_i_end = curves(i).points(end, :);
    
    for j = 1:num_curves
        if i ~= j
            curve_j_start = curves(j).points(1, :);
            
            % Calculate distance between end of curve i and start of curve j
            distance = norm(curve_i_end - curve_j_start);
            
            % If distance is below threshold, mark these curves to be joined
            if distance < distance_threshold
                connection_graph(i, j) = 1;
                fprintf('Joining curve %d to curve %d (distance: %.6f)\n', curves(i).id, curves(j).id, distance);
            end
        end
    end
end

%% Identify connected components using graph analysis
% This represents groups of curves that should be joined
joined_curves = {};
visited = false(1, num_curves);

% Define DFS function for finding connected components
% Note: In MATLAB, nested functions need to be at the end of the script
% or defined separately. We'll implement the DFS algorithm directly here.

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
                    if connection_graph(node, neighbor) && ~visited(neighbor)
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

% Handle standalone curves (no connections)
for i = 1:num_curves
    if ~visited(i)
        joined_curves{end+1} = i;
    end
end

fprintf('After joining, found %d curve groups\n', length(joined_curves));

%% Create figure
figure;
hold on;

% Define a fixed set of 20 distinct colors that will repeat
fixed_colors = jet(30);

% Count statistics
total_curves = 0;
skipped_curves = 0;

%% Plot each joined curve group
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
            skipped_curves = skipped_curves + 1;
            continue;
        end
    else
        % For joined curves, concatenate points in the correct order
        % We need to determine the correct ordering based on connection_graph
        ordered_indices = [];
        remaining = group;
        
        % Find a curve with no incoming edges to start
        start_idx = -1;
        for idx = group
            if ~any(connection_graph(:, idx))
                start_idx = idx;
                break;
            end
        end
        
        % If no clear start found, just use the first curve
        if start_idx == -1
            start_idx = group(1);
        end
        
        ordered_indices = [ordered_indices, start_idx];
        remaining(remaining == start_idx) = [];
        
        % Build the path by following connections
        current = start_idx;
        while ~isempty(remaining)
            found_next = false;
            for idx = remaining
                if connection_graph(current, idx)
                    ordered_indices = [ordered_indices, idx];
                    remaining(remaining == idx) = [];
                    current = idx;
                    found_next = true;
                    break;
                end
            end
            
            % If no next curve found but remaining curves exist, 
            % this means we have a disconnected subgraph
            if ~found_next && ~isempty(remaining)
                ordered_indices = [ordered_indices, remaining(1)];
                current = remaining(1);
                remaining(1) = [];
            end
        end
        
        % Now concatenate points in the determined order
        points = [];
        directions = [];
        for j = 1:length(ordered_indices)
            idx = ordered_indices(j);
            points = [points; curves(idx).points];
            directions = [directions; curves(idx).directions];
        end
        
        % Skip if joined curve is too small
        if size(points, 1) < min_curve_size
            fprintf('Skipping joined curve group %d with only %d points (minimum is %d)\n', ...
                i, size(points, 1), min_curve_size);
            skipped_curves = skipped_curves + 1;
            continue;
        end
    end
    
    % Use modulo to cycle through the 30 colors
    color_idx = mod(i-1, 30) + 1;
    color = fixed_colors(color_idx, :);
    
    % Plot the curve
    plot3(points(:, 1), points(:, 2), points(:, 3), '-', 'LineWidth', 2, 'Color', color);
    quiver3(points(:,1), points(:,2), points(:,3), mag*directions(:,1), mag*directions(:,2), mag*directions(:,3), 0, 'Color', 'k', 'LineWidth', 0.5, 'MaxHeadSize', 0.5);
    
    total_curves = total_curves + 1;
    fprintf('Plotted curve group %d with %d points\n', i, size(points, 1));
end

fprintf('Plotted %d curves, skipped %d curves with fewer than %d points\n', ...
    total_curves, skipped_curves, min_curve_size);

title('3D Curves from Connectivity Graph with Joined Curves');
axis off;
axis equal;
view(3); 
set(gcf, 'color', 'w');
ax = gca;
ax.Clipping = "off";