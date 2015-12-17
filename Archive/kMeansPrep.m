function [ ret ] = kMeansPrep( data )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    [n,~] = size(data);
    
    cell_size = 8;
    [x,y,z] = size(vl_hog(single(getImage(data, 1)), cell_size));
    %hog = zeros(4,n);
    hog = zeros(x*y*z,n);
    domi = zeros(3,n);
    for i=1:n
       img = single(getImage(data, i));
       hog(:,i) = reshape(vl_hog(img, cell_size),[x*y*z,1]);
       %domi(:,i) = dominantColor(data,i)./200;
       %hog(4,i) = 7 * edgeStrength( data, 'sobel', i );
    end
    %ret = [hog;domi];
    ret = hog;
    disp('end of prep');
end

