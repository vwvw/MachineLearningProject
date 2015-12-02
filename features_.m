function [ feat ] = features( data )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    [n,~] = size(data);
    
    
    hog = zeros(n, 7936);
    
    for i=1:n
       image = single(getImage(data, i));
       hog(i,:) = reshape(vl_hog(image, 2), 1, []);
    end

    feat = hog;

end

