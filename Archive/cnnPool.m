function [poold] = cnnPool (convd, poolsize, gpu_accel)
% Performs a pooling layer.

    [m,~,F,N] = size(convd);
    
    k = m/poolsize;
    
    if gpu_accel
        poold = zeros(k, k, F, N, 'gpuArray');
    else
        poold = zeros(k, k, F, N);
    end
    
    for n=1:N
        for f=1:F
            for i=1:k
                for j=1:k
                    poold(i,j,f,n) = ...
                        max(max(convd((i-1)*poolsize+1:i*poolsize, ...
                                    (j-1)*poolsize+1:j*poolsize, f, n)));
                end
            end
        end
    end
                                                    

end