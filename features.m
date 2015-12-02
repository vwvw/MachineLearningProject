function [ feat ] = features( data )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    [n,~] = size(data);
    
    
    hog = zeros(n, 279);
    
    for i=1:n
       image = single(getImage(data, i));
       hog(i,:) = reshape(vl_hog(image, 12), 1, 279);
    end

    feat = hog;

end

