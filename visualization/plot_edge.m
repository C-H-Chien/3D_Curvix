clear;
close all;

% === Settings ===
validation_view_index = 48;
Dataset_Path = "/gpfs/data/bkimia/Datasets/ABC-NEF/";
Object_Name = "00000006/";
Image_Path = fullfile(Dataset_Path, Object_Name, "train_img/");
img_filename = fullfile(Image_Path, sprintf('%d_colors.png', validation_view_index));
img = imread(img_filename);
mag = 10;

% === Cluster coordinates ===
edges_1 = [ 
  341.865 420.017 1.28181
]; 
edges_2 = [ 
 340.811 415.868 1.39218
];

% === Plot ===
plot_handles = [];
figure; imshow(img); hold on;
h1 = plot(edges_1(:,1),  edges_1(:,2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
plot_handles = [plot_handles, h1];
u_after = mag * cos(edges_1(:,3));
v_after = mag * sin(edges_1(:,3));
quiver(edges_1(:,1), edges_1(:,2), u_after, v_after, 0, 'r', 'LineWidth', 2);hold on;

h2 = plot(edges_2(:,1),  edges_2(:,2), 'rx', 'LineWidth', 2, 'MarkerSize', 10);
plot_handles = [plot_handles, h2];
u_after = mag * cos(edges_2(:,3));
v_after = mag * sin(edges_2(:,3));
quiver(edges_2(:,1), edges_2(:,2), u_after, v_after, 0, 'r', 'LineWidth', 2);

legend([h1, h2], {'Edge 1', 'Edge 2'}, 'Location', 'northwest');

