function [reshaped] = cnnUnrollPool (poold, k, f, gpu_accel)

    [n,~] = size(poold);
    
    if gpu_accel
        reshaped = zeros(k,k,f,n,'gpuArray');
    else
        reshaped = zeros(k,k,f,n);
    end
    
    for i=1:n
        reshaped(:,:,:,i) = reshape(poold(i,:), k, k, f);
    end

end