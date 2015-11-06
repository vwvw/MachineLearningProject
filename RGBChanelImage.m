function [R,G,B ] = RGBChanelImage( data, i )
%Output 3 matrices, each one representing one image with only one chanel of the RGB image
%1st matrix : red values, 
%2nd matrix : green values,
%3rd matrix : blue values,
im = getImage(data,i);
z = zeros(32,32);
R = cat(3, im(:,:,1), z,z);
G = cat(3, z,im(:,:,1),z);
B = cat(3, z,z,im(:,:,1));


end

