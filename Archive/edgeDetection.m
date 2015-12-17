function [  ] = edgeDetection( data,method, i  )
% Edge detection of image i in data "data"
A = getImage(data,i);

%gray scale edge detection
figure;
subplot(2,4,1), subimage(A);
title('original image');
BW = rgb2gray(A);
subplot(2,4,2), imshow(BW);
title('image Black & White');
for j = 1 : 4
    BWEdge =  edgeGrayScale(A,method, 0.05*j);
    subplot(2,4,4+j), subimage(BWEdge);
    str = ['treshold = ', num2str(j*0.05)];
    title(str);
end


%RGB edge detection
figure
subplot(2,4,1), subimage(A);
title('original image');
[R,G,B] = RGBChanelImage(data,i);
subplot(2,4,2), subimage(R);
title('red chanel');
subplot(2,4,3), subimage(G);
title('green chanel');
subplot(2,4,4), subimage(B);
title('blue chanel');
treshhold = 0.05;
edgeR = edge(rgb2gray(R), method, treshhold);
edgeG = edge(rgb2gray(G), method, treshhold);
edgeB = edge(rgb2gray(B), method, treshhold);
edgeTotal = edgeR | edgeG | edgeB;
subplot(2,4,5), imshow(edgeTotal);
title('edge total');
subplot(2,4,6), imshow(edgeR);
title('edge on red');
subplot(2,4,7), imshow(edgeG);
title('edge on green');
subplot(2,4,8), imshow(edgeB);
title('edge on blue');

edgeStrength(data, method, i)

end

