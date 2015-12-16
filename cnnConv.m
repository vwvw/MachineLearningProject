function [ convd ] = cnnConv (data, W, b, gpu_accel)
% Performs a convolution layer on data given weight and bias.
% If gpu_accel is not set, data, W, b are expected to be matrices,
% convd is returned as a matrix.
% If gpu_accel is set, data, W, b are expected to be gpuArrays,
% convd is returned as a gpuArray.
%
% Parameters:
%   data - incoming layer : m x m x n, m - image dimension, n - num images
%   W    - filter weight  : k x k x f, k - filter dimension, f - num
%   filters
%   b    - bias : f x 1 - filter bias
%
% Returns
%   convd - c x c x f x n - c - convolved dimension = m - k + 1

    [m, ~, n] = size(data);
    [k, ~, f] = size(W);
    
    
    convDim = m - k + 1;
    
    if gpu_accel
        convd = zeros(convDim, convDim, f, n, 'gpuArray');
    else
        convd = zeros(convDim, convDim, f, n);
    end
    
    
    for i=1:n
       for j=1:f
           img = data(:,:,i);
           filter = rot90(W(:,:,j),2);
           convd(:,:,j,i) = nnLayer(1, conv2(img, filter, 'valid'), b(j));
       end
    end

end