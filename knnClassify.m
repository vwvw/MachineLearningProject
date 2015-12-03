function [ output ] = knnClassify( data, km, k, labels)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

    [n,~] = size(data);
    
    hog = kMeansPrep(data, zeros(n,1));
    
    [~,~,~,m] = size(km);
    
    output = zeros(n,1);
    
    parfor i=1:n
        means = -ones(k,2);
        
        for j=1:m
            
            %compute the distance function
            delta = sum(sum(sum(abs(hog(:,:,:,i)-km(:,:,:,j)))));
            
            
            for p=1:k
                if delta < means(p,1) || means(p,1) == -1
                    means(p+1:end,:) = means(p:end-1,:);
                    means(p,1) = delta;
                    means(p,2) = labels(j);
                    break;
                end
            end
            
        end
        output(i) = mode(means(:,2));
    end


end

