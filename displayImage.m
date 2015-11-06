function [  ] = displayImage( data, i )
%UNTITLED2 Displaying an image of Cifar 10
%   Detailed explanation goes here
A = getImage(data,i);
% Display the color image.
image(A);
axis on;
end

