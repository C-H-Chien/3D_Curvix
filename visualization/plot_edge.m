
clear;
clc;
close all;

fx = 1111.11136542426;
fy = 1111.11136542426;
cx = 399.500000000000;
cy = 399.500000000000;
K = [fx, 0, cx; 0, fy, cy; 0, 0, 1];
invK = inv(K);

%> Function Macro for constructing a skew-symmetric matrix
skew_T = @(T)[0, -T(3,1), T(2,1); T(3,1), 0, -T(1,1); -T(2,1), T(1,1), 0];


R1 = [0.577608, -0.214887, 0.787523; -0.816314,  -0.15205,  0.557236; 4.91473e-09,  0.96473,  0.263241]';%FRAME 0
R2 = [0.515038, 0.361468, 0.777224; -0.857168, 0.217192, 0.467003; -5.04312e-08,     0.906735,    -0.421701]';
T1 = [0.119353; -0.298896; -3.98703]; %FRAME 0
T2 = [0.171065; -0.742698; -3.92672];
point1 = [557.268; 414.491; 1];%FRAME 0
point2 = [544.67; 349.49; 1];


img_file = "/gpfs/data/bkimia/Datasets/ABC-NEF/00000006/train_img/5_colors.png";
imshow(img_file);
img2 = imread(img_file);
axis on;
hold on;
show_lengeng = "Frame 5";
plot(point2(1), point2(2), 'ro', 'MarkerSize', 10);  % Use 'ro' for a red circle marker and adjust MarkerSize
title('Frame 5');   
hold on;


Rel_R = R2 * R1';
Rel_T = T2 - R2 * R1' * T1;
E21 = skew_T(Rel_T) * Rel_R;
F21 = invK' * E21 * invK;
l = F21 * point1;

imageWidth = size(img2, 2);
x = [1, imageWidth]; 
y = (-l(1) * x - l(3)) / l(2); % Y-coordinates

plot(x, y, 'r', 'LineWidth', 2);
hold off;




