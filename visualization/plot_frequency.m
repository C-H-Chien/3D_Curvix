clear;
close all;

filePath = '/gpfs/data/bkimia/zqiwu/3D/3D_Edge_Sketch_and_Grouping/outputs/paired_edges_final_0_40.txt';

fileID = fopen(filePath, 'r');
data = textscan(fileID, '%s', 'Delimiter', '\n', 'Whitespace', '');
fclose(fileID);

lines = data{1};
pairRowCounts = [];
currentRowCount = 0;

for i = 1:length(lines)
    if isempty(lines{i}) % Check for blank line separating pairs
        if currentRowCount > 0
            pairRowCounts = [pairRowCounts, currentRowCount-1];
            currentRowCount = 0;
        end
    else
        currentRowCount = currentRowCount + 1;
    end
end

if currentRowCount > 0
    pairRowCounts = [pairRowCounts, currentRowCount];
end

figure;
histogram(pairRowCounts);
xlabel('Number of Supporting Validation Views');
ylabel('Frequency');
title('Distribution of Number of Supporting Validation Views for Hypothesis View 0 and 40');
