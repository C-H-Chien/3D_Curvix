function Error = getProjectionError(Points_3D, Points_2D, K, x)

    eul_angle = x(1:3);
    transl    = x(4:6);

    R = eul2rotm(eul_angle);
    T = transl';

    projection = K * (R * Points_3D + T);
    projection = projection ./ projection(3,:);

    reproj_errs = vecnorm(projection(1:2,:) - Points_2D(1:2,:), 2, 1);
    
    % Error = sum(reproj_errs) / length(reproj_errs);
    Error = sum(reproj_errs);
end