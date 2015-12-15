function [ tree ] = knn_construct_kdTree( points,d )
%Construct a kdTree from points along the dimension d
[~,n] = size(points);
if n >2
    [~, index] = sort(points(d,:));
    points_sorted = points(:,index);
    if mod(n,2) == 1
        smaller_points = points_sorted(:,1:(n-1)/2);
        node = points_sorted(:,(n-1)/2+1);
        bigger_points = points_sorted(:,(n-1)/2+2:end);
    else
        smaller_points = points_sorted(:,1:n/2);
        node = points_sorted(:,n/2+1);
        bigger_points = points_sorted(:,n/2+2:end);
    end

    left = knn_construct_kdTree(smaller_points,d+1);
    right = knn_construct_kdTree(bigger_points,d+1);
    tree = struct('node',node,'left_tree', left, 'right_tree', right);
elseif n == 2
   tree = struct('node', points(:,2), 'left_tree', points(:,1), 'right_tree', []);      
elseif n == 1
   tree = struct('node', points(:,1), 'left_tree', [], 'right_tree', []); 
else
    disp('error')
end
end

