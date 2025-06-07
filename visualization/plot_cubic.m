data = [
    182.2338, 204.5992, 196.7061, 194.6796, 202.3188, 196.0671, 200.5113, 196.0775, 197.7951, 202.9657, 188.3827, 196.2408;
    179.7696, 156.9738, 196.2999, 193.3563, 157.3916, 196.3148, 198.2328, 164.2800, 197.3030, 202.0402, 170.1162, 196.5253;
    202.8114, 188.3345, 196.2539, 207.8706, 178.3031, 196.9117, 199.5269, 167.2161, 196.1547, 201.9868, 170.1513, 196.5420;
    157.1873, 185.2353, 196.6378, 158.1357, 192.7082, 196.7832, 163.5174, 199.1683, 196.7007, 174.2570, 204.0027, 196.6973;
    158.2663, 171.5599, 196.7205, 165.3533, 158.5868, 197.5515, 170.9646, 158.1020, 196.9641, 179.7426, 156.9789, 196.1701
];
% Assume 'data' is already loaded with N x 12 format

% Create a new figure
figure;

% Set up the figure properties
hold on;
view(3);
set(gcf, 'color', 'w');
ax = gca;
ax.Clipping = "off";
axis off;
axis equal;

% Loop through each row and plot the polyline through 4 points
for i = 1:size(data, 1)
    % Extract coordinates for the four points
    x = [data(i, 1), data(i, 4), data(i, 7), data(i, 10)];
    y = [data(i, 2), data(i, 5), data(i, 8), data(i, 11)];
    z = [data(i, 3), data(i, 6), data(i, 9), data(i, 12)];
    
    % Plot the polyline as three segments: 1->2->3->4
    plot3(x, y, z, 'LineWidth', 1.5, 'Color', [0.3, 0.3, 0.8]);
end

hold off;
