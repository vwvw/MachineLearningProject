function [ t ] = reshapeLabels( labels, lsize, gpu_accel)
%reshapeLabels Reshapes the labels in terms of activator values

    [n,~] = size(labels);
    t = zeros(n, lsize);
    labels = labels + 1;
    
    if gpu_accel
        t = gpuArray(t);
    end
    
    for i=1:n
       t(i, labels(i)) = 1; 
    end
end

