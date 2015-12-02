function [ data ] = cnnReshapeData( data_d, gpu_accel ) =
% Reshapes the data from a nx(32*32*3) matrix to a 32x32xn matrix

    [n,~] = size(data_d);
    
    data = zeros(32, 32, n);
    
    for i=1:n
        img = getImage(data_d, i);
        data(:,:,i) = single(rgb2gray(img));
    end
    
    
    %% if gpu_acceleration is defined, cast the image into a gpuArray
    if gpu_accel
        data = gpuArray(data);
    end

end