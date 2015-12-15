function [ output ] = knnClassify( data, training_tree, k)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

    [n,~] = size(data);
    
    hog = kMeansPrep(data);
    [dim,~] = size(hog);
    output = zeros(n,1);

    for i = 1:n
       % iterating for each points to find the k nearest poitns
        %first column dist, last column label
        BPQ = knn_find_k_closest( hog(:,i)', training_tree, Inf(k,dim+2),1);
        output(i) = mode(BPQ(:,end));
        disp(i)
    end
    
    
    
    
    
    %% old code
%     parfor i=1:n
%         
%         means = Inf(k,2);
%         if mod(i,100) == 0
%             disp(i);
%         end
%         for j=1:m
%             
%             %compute the distance function
%             delta = sum(abs(hog(:,i)-km(:,j)));
%             
%             if delta < means(1,1) 
%                 p = 1;
%                 while p <k && means(p+1,1)>delta
%                     p = p+1;
%                 end
%                 if p ~= 1
%                     means(1:p-1, :) = means(2:p,:);
%                 end
% 
% 
%                 means(p,1) = delta;
%                 means(p,2) = labels(j);
% 
%             end
%             
%         end
%         output(i) = mode(means(:,2));
%     end


end

