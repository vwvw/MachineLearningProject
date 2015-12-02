function [ c ] = correctness( output, labels )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    [n,~] = size(output);
    
    c = 0;
    
    for i=1:n
       
        if(output(i)==labels(i))
          c = c + 1;
        end
        
    end
    
    c = c / n * 100;

end

