close all;clear;

idx = 49;
figure(1);
for i = 0:idx
    % figure(i+1);
    plotEdgeIterationForces(i);
    pause(0.5);
    % if i == 0, waitforbuttonpress; end
    % waitforbuttonpress;
    hold off;
end
