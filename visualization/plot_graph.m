% Initialize figure
figure;
hold on;
axis equal;

% Edge locations (x, y, z)
edge_locations = [
    0.703582, 0.881398, 0.353133;  % Edge 1
    0.710306, 0.877993, 0.353316;  % Edge 2
    0.704316, 0.884627, 0.35257;   % Edge 3
    0.710307, 0.881131, 0.352345;  % Edge 4
    0.675811, 0.901782, 0.353594;  % Edge 5
    0.678757, 0.900492, 0.353766   % Edge 6
];

% Edge orientations (not used for plotting positions, but could be used for visualization)
edge_orientations = [
    -0.865327, 0.501203, 0.00209428;   % Edge 1
    -0.868224, 0.496146, -0.00518501;  % Edge 2
    0.85584, -0.517224, 0.00410801;    % Edge 3
    0.853602, -0.520643, -0.0171691;   % Edge 4
    -0.857672, 0.514105, 0.00973868;   % Edge 5
    -0.836825, 0.544941, 0.0525654     % Edge 6
];

% Plot each edge point
scatter3(edge_locations(:,1), edge_locations(:,2), edge_locations(:,3), 100, 'filled');

% Edge connections based on the graph
connections = [
    1, 2;
    1, 3;
    1, 4;
    2, 3;
    2, 4;
    3, 4;
    3, 6;
    4, 5;
    4, 6;
    5, 6
];

% Plot connections
for i = 1:size(connections, 1)
    p1 = edge_locations(connections(i,1),:);
    p2 = edge_locations(connections(i,2),:);
    plot3([p1(1), p2(1)], [p1(2), p2(2)], [p1(3), p2(3)], 'k-', 'LineWidth', 1.5);
end

% Add edge labels
for i = 1:size(edge_locations, 1)
    text(edge_locations(i,1), edge_locations(i,2), edge_locations(i,3) + 0.002, ...
        ['Edge ', num2str(i)], 'FontSize', 12, 'HorizontalAlignment', 'center');
end

% Set the view angle
view(45, 30);

% Add grid and labels
grid on;
set(gcf, 'color', 'w');
ax = gca;
ax.Clipping = "off";
hold off;
title('3D Edge Graph');

% Add a legend
legend('Edge Points', 'Connections');