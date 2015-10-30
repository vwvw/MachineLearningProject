function [ output_args ] = displayImage( data, i )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
R=data(i,1:1024);
G=data(i,1025:2048);
B=data(i,2049:3072);
% Create a 32x32 color image.
A = zeros(32,32,3, 'uint8');
A(:,:,1)=reshape(R,32,32);
A(:,:,2)=reshape(G,32,32);
A(:,:,3)=reshape(B,32,32);
% Display the color image.
image(A);
axis on;

end

