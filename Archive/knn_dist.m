function [ dist ] = knn_dist( point1, point2 )
    dist = sum(abs(point1-point2));


end

