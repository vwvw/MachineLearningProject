function [ maxH ] = horizonLine( data, i )

A = getImage(data,i);
l = size(A,1);
edge = edgeGrayScale(A, 'sobel',0.14);
subimage(edge);
h = zeros(l-2,1);
for i = 2 : l-1
    h(i,1) = sum(edge(i+1,:))+sum(edge(i,:))+sum(edge(i-1,:));
end
h = sort(-h);
maxH = -h(1)+h(4); % since we take the number of edges on 3 lines,
                %we look at the 4th group of line with the highest number of edges.


end

