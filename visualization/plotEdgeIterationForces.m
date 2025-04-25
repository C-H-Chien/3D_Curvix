function plotEdgeIterationForces(iteration)
    % Function to plot edge location, orientation, and forces for a specific iteration
    % Input: iteration - the iteration number to plot (0-indexed)
    
    %> Define the folder path containing the 3D edge output files
    data_folder_name = 'outputs';
    data_folder_path = fullfile(fileparts(mfilename('fullpath')), '..', data_folder_name);
    
    % Load the data files
    before_alignment_path = fullfile(data_folder_path, "test_3D_edges_before_smoothing.txt");
    after_alignment_path = fullfile(data_folder_path, "test_3D_edges_after_smoothing.txt");
    forces_path = fullfile(data_folder_path, "test_3D_edges_total_forces.txt");
    
    before_alignment = importdata(before_alignment_path);
    after_alignment = importdata(after_alignment_path);
    
    % Calculate number of points and verify iteration is valid
    num_of_points = size(before_alignment, 1);
    num_of_iters = size(after_alignment, 1) / num_of_points;
    
    if iteration < 0 || iteration >= num_of_iters
        error('Iteration number out of range. Valid range: 0 to %d', num_of_iters-1);
    end
    
    % Load force data
    force_vectors = [];
    if exist(forces_path, 'file')
        % Open the file and read all content
        force_content = fileread(forces_path);
        % Split by new line to get each iteration
        force_lines = regexp(force_content, '\n', 'split');
        
        % Check if we have enough iterations
        if iteration < length(force_lines)
            force_line = force_lines{iteration+1}; % +1 because iteration is 0-indexed
            
            % Parse the line by semicolons
            force_parts = regexp(force_line, ';', 'split');
            
            force_vectors = zeros(length(force_parts), 3);
            
            for i = 1:length(force_parts)
                if ~isempty(strtrim(force_parts{i}))
                    % Parse the force components
                    force_str = strtrim(force_parts{i});
                    components = sscanf(force_str, '%f %f %f');
                    
                    if length(components) == 3
                        force_vectors(i,:) = components';
                    end
                end
            end
        else
            warning('Force data for iteration %d not found', iteration);
        end
    else
        % warning('Force file not found: %s', forces_path);
    end
    
    % Define color coding for points
    color_code = [
        1.0 0.0 0.0;    % red
        0.0 0.0 1.0;    % blue
        0.0 0.8 0.0;    % green
        0.0 1.0 1.0;    % cyan
        1.0 0.0 1.0;    % magenta
        1.0 1.0 0.0;    % yellow
        0.0 0.0 0.0;    % black
        1.0 0.5 0.0;    % orange
        0.5 0.0 0.5;    % purple
        0.0 0.5 0.0;    % dark green
        0.5 0.5 0.0;    % olive
        0.0 0.5 1.0;    % sky blue
        0.7 0.3 0.3;    % brown
        0.5 0.5 0.5     % gray
    ];
    
    % Set magnitude scaling factors
    mag_orientation = 0.001;  % for orientation vectors
    mag_force = 0.5;        % for force vectors
    
    % Create figure
    % figure;
    
    % Get the indices for the requested iteration
    start_idx = iteration * num_of_points + 1;
    
    % Plot each point for the given iteration
    for i = 0:num_of_points-1
        % Get correct color index with wraparound
        color_idx = mod(i, size(color_code, 1)) + 1;
        
        % Get the point index in the after_alignment array
        point_idx = start_idx + i;
        
        % Get the current point location
        point_loc = after_alignment(point_idx,1:3);
        
        % Get the orientation vector
        orientation_vec = after_alignment(point_idx,4:6);
        
        % Plot the point location
        plot3(point_loc(1), point_loc(2), point_loc(3), ...
            'Color', color_code(color_idx,:), 'Marker', 'o', 'MarkerSize', 5, 'MarkerFaceColor', color_code(color_idx,:)); 
        hold on;
        
        % Plot orientation vector as a line through the point
        quiver3(point_loc(1), point_loc(2), point_loc(3), ...
                   mag_orientation*orientation_vec(1), mag_orientation*orientation_vec(2), mag_orientation*orientation_vec(3), ...
                   0, 'LineWidth', 1, 'Color', 'r', 'MaxHeadSize', 0.5);
        % line([point_loc(1) + mag_orientation*orientation_vec(1), point_loc(1) - mag_orientation*orientation_vec(1)], ...
        %      [point_loc(2) + mag_orientation*orientation_vec(2), point_loc(2) - mag_orientation*orientation_vec(2)], ...
        %      [point_loc(3) + mag_orientation*orientation_vec(3), point_loc(3) - mag_orientation*orientation_vec(3)], ...
        %      'Color', color_code(color_idx,:), 'LineWidth', 1); 
        % 
        % Plot force vector if available
        if ~isempty(force_vectors) && i+1 <= size(force_vectors, 1)
            force_vec = force_vectors(i+1,:);
            
            % Calculate start and end points for the force arrow
            force_start = point_loc;
            force_end = point_loc + mag_force * force_vec;
            
            % Plot force vector as an arrow using quiver3
            quiver3(force_start(1), force_start(2), force_start(3), ...
                   mag_force*force_vec(1), mag_force*force_vec(2), mag_force*force_vec(3), ...
                   0, 'LineWidth', 1, 'Color', 'g', 'MaxHeadSize', 0.5);
        end
    end
    
    % Set plot properties
    axis equal;
    % title(sprintf('Edge Locations, Orientations, and Forces at Iteration %d', iteration));
    title(sprintf('Edge Locations and Orientations at Iteration %d', iteration));
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    grid on;

    
    % Set the background color to white
    set(gcf, 'color', 'w');
    ax = gca;
    ax.Clipping = "off";

    
    hold off;
end