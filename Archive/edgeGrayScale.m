function [ BWEdge ] = edgeGrayScale(image, method, treshold )

BW = rgb2gray(image);
BWEdge =  edge(BW ,method, treshold);
end

