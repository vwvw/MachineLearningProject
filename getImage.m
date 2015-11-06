function [ image ] = getImage( data, i )
%Return the image i in the data set "data"
R=data(i,1:1024);
G=data(i,1025:2048);
B=data(i,2049:3072);
% Create a 32x32 color image.
image = zeros(32,32,3, 'uint8');
image(:,:,1)=reshape(R,32,32);
image(:,:,2)=reshape(G,32,32);
image(:,:,3)=reshape(B,32,32);
%Rotate it in the corect direction
image = rot90(image,3);
end

