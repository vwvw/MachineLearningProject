function [reshaped] = cnnReshapePool (poold, gpu_accel)

    [k,~,f,n] = size(poold);
    
    if gpu_accel
        reshaped = zeros(n,k*k*f, 'gpuArray');
    else
        reshaped = zeros(n,k*k*f);
    end
    
    for i=1:n
        reshaped(i,:) = reshape(poold(:,:,:,i), 1, []);
    end

end