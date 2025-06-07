clear;
close all;

image_num = 32;
% Define dataset and image paths
Dataset_Path = "/gpfs/data/bkimia/Datasets/ABC-NEF/";
Object_Name = "00000006/";
Image_Path = fullfile(Dataset_Path, Object_Name, "train_img");
Edges_Path = fullfile(Dataset_Path, Object_Name, "Edges");
Edgel_file = importdata(strcat(Edges_Path, "/Edge_", string(image_num), "_t1.txt"));

figure;
img_path = sprintf('/gpfs/data/bkimia/Datasets/ABC-NEF/00000006/train_img/%d_colors.png',  image_num);
img = imread(img_path);
imshow(img); hold on;

x_red = 309.375;
y_red = 339.496;

mag = 1;

x_blue = 309.417;
y_blue = 337.996;
% cos_theta_blue = 0.0419301; 
% sin_theta_blue = 0.999121;
% 
% vec1 = [0.163725	0.168102	0.972078];
% vec2 = [-0.0126648	0.00919256	0.999878];
plot(Edgel_file(:,1),  Edgel_file(:,2), 'co', 'MarkerSize', 10);hold on;

mag = 0.5;
theta = Edgel_file(:,3); % Assuming angles are in radians
u = cos(theta) * mag;    % x component of the arrow
v = sin(theta) * mag;    % y component of the arrow

% Plot arrows for all edges
quiver(Edgel_file(:,1), Edgel_file(:,2), u, v, 0, 'g-', 'LineWidth', 1.2);

plot(x_red,  y_red, 'ro', 'MarkerSize', 10);hold on;
%plot([x_red + mag*cos_theta, x_red - mag*cos_theta], [y_red + mag*sin_theta, y_red - mag*sin_theta],'r-', 'LineWidth', 1.2);hold on;

plot(x_blue, y_blue, 'bo', 'MarkerSize', 10);hold on;
%plot([x_blue + mag * cos_theta_blue, x_blue-mag*cos_theta_blue], [y_blue+mag*sin_theta_blue, y_blue-mag*sin_theta_blue],'b-', 'LineWidth', 1.2);



