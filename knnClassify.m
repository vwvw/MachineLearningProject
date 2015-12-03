function [ output ] = knnClassify( data, km, k, labels)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

    [n,~] = size(data);
    
    hog = kMeansPrep(data);
    [~,m] = size(km);
    output = zeros(n,1);
    
    parfor i=1:n
        
        means = Inf(k,2);
        if mod(i,100) == 0
            disp(i);
        end
        for j=1:m
            
            %compute the distance function
            delta = sum(abs(hog(:,i)-km(:,j)));
            
            if delta < means(1,1) 
                p = 1;
                while p <k && means(p+1,1)>delta
                    p = p+1;
                end
                if p ~= 1
                    means(1:p-1, :) = means(2:p,:);
                end


                means(p,1) = delta;
                means(p,2) = labels(j);

            end
            
        end
        output(i) = mode(means(:,2));
    end


end

