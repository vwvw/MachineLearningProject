function [ feat ] = features( data )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    [n,~] = size(data);
    
    %edge = zeros(n, 1024);
    %domColors = zeros(n, 3);
    %hori = zeros(n,1);
    hog = zeros(n, 279);
    
    for i=1:n
       image = single(getImage(data, i));
     %  domColors(i,:) = dominantColor(data, i);
       hog(i,:) = reshape(vl_hog(image, 12), 1, 279);
      % hori(i) = horizonLine(data,i);
       %edge(i,:) = reshape(edgeGrayScale(image, 'sobel', 0.12), 1,1024);
    end
    feat = hog;
    %feat = horzcat(domColors./255, hog, hori, edge);

end

