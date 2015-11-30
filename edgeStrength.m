function [ strength ] = edgeStrength( data, method, i )
%ouput the percentage of element edges still present with a treshold of .2
%in comparison with a treshold of .15
A = getImage(data,i);
treshold1 = 0.15;
treshold2 = 0.2;
edge1 = edgeGrayScale(A, method, treshold1);
edge2 = edgeGrayScale(A, method, treshold2);
%figure
%subplot(1,2,1), subimage(edge1);
%subplot(1,2,2), subimage(edge2);
nbrEdge1 = sum(sum(edge1));
nbrEdge2 = sum(sum(edge2));
strength  = nbrEdge2/nbrEdge1* 100;
end