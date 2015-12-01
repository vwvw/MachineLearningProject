function [ feat ] = features( data )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    [n,~] = size(data);
    
    %edge = zeros(n, 1024);
    domColors = zeros(n, 3);
    
    hog = zeros(n, 496);
    
    
    %% compute the first 10 SIFT features
    
    for i=1:n
       image = single(getImage(data, i));
       domColors(i,:) = dominantColor(data, i);
       hog(i,:) = reshape(vl_hog(image, 8), 1, 496);
    end

    feat = horzcat(domColors, hog);

end

