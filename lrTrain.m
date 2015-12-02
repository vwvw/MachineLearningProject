function [ w , error] = lrTrain( data, labels, rate, iter, gpu_accel)
%lrTrain Logistic Regression training
    
    [N, D] = size(data);
    [~, L] = size(labels);
    w = rand(D,L) * 10;
    error = zeros(1,iter);
    
    
    if gpu_accel
        w = gpuArray(w);
        error = gpuArray(error);
    end
    
    for i=1:iter
       p = labels - sigmoid(1,data,w);
       w = w + rate * (transpose(data) * p);
       error(i) = sum(sum(abs(p)));
    end
    
    if gpu_accel
        error = gather(error);
    end
    
    plot(error)
end

