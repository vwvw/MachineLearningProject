function [ x ] = lrFeatures(data)
%lrFeatures Convert image RGB values to HSV values
%   Detailed explanation goes here

    [N,~] = size(data);

    pixel = rgb2hsv(reshape(data, N, [], 3));
    
    %use value component of HSV
    x = horzcat( ones(N, 1), pixel(:,:,3));
end