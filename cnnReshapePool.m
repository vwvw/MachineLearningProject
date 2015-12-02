function [reshaped] = cnnReshapePool (poold, gpu_accel)

    [k,~,f,n] = size(poold);
    
    reshaped = zeros(n, k*k*f);
    
    if gpu_accel
        reshaped = gpuArray(reshaped);
    end
    
    for i=1:n
        reshaped(i,:) = reshape(poold(:,:,:,i), 1, []);
    end

end