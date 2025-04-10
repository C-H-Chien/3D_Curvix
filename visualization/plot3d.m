% Given 3D points and tangents
points = [
    -0.374895, 0.333224, 0.152513;
    -0.374246, 0.333140, 0.153784;
    -0.373580, 0.333050, 0.155058;
    -0.373807, 0.333083, 0.155370;
    -0.373584, 0.333054, 0.156164;
    -0.376447, 0.333420, 0.148066;
    -0.375782, 0.333329, 0.149340;
    -0.375554, 0.333301, 0.150134;
    -0.375329, 0.333274, 0.150927;
    -0.374676, 0.333191, 0.152200
];

tangents = [
    0.875440, -0.100009, 0.472868;
    0.875455, -0.100184, 0.472803;
    0.875469, -0.100357, 0.472740;
    0.875424, -0.100415, 0.472811;
    0.875409, -0.100530, 0.472815;
    0.875485, -0.0993751, 0.472918;
    0.875499, -0.0995479, 0.472855;
    0.875485, -0.0996628, 0.472859;
    0.875470, -0.0997779, 0.472862;
    0.875485, -0.099952, 0.472797
];

% Scale for tangent vector length
mag = 0.0025;

% Plot the 3D points and tangents
figure;
hold on;
grid on;
axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Points and Tangents');

% Plot each point and its tangent vector
for j = 1:size(points, 1)
    P = points(j, :)'; % Current edge point
    T_v = tangents(j, :)'; % Corresponding tangent vector

    % Plot the 3D point
    plot3(P(1), P(2), P(3), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

    % Plot the tangent vector as a line
    line([P(1) + mag * T_v(1), P(1) - mag * T_v(1)], ...
         [P(2) + mag * T_v(2), P(2) - mag * T_v(2)], ...
         [P(3) + mag * T_v(3), P(3) - mag * T_v(3)], ...
         'Color', 'k', 'LineWidth', 0.5, 'HandleVisibility', 'off'); % Exclude from legend
end

legend('3D Points');
view(3);
hold off;
