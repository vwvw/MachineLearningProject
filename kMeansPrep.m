function [ hog ] = kMeansPrep( data, label )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    [n,~] = size(data);
    
    cell_size = 8;
    [x,y,z] = size(vl_hog(single(getImage(data, 1)), cell_size));
    hog = zeros(x,y,z,n);
    
    
    parfor i=1:n
       img = single(getImage(data, i));
       hog(:,:,:,i) = vl_hog(img, cell_size);
    end

end

