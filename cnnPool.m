function [poold] = cnnPool (convd, poolsize, gpu_accel)
% Performs a pooling layer.

    [m,~,F,N] = size(convd);
    
    k = m/poolsize;
    
    poold = zeros(k, k, F, N);
    
    if gpu_accel
        poold = gpuArray(zeros);
    end
    
    for n=1:N
        for f=1:F
            for i=1:k
                for j=1:k
                    poold(i,j,f,n) = ...
                        mean2(convd((i-1)*poolsize+1:i*poolsize, ...
                                    (j-1)*poolsize+1:j*poolsize, f, n));
                end
            end
        end
    end
                                                    

end