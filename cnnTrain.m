function [ w, b, w_i, w_o ] = cnnTrain( ...
    trainingData, ... 
    labels, ...
    lsize, ...
    poolsize, ...
    filtersize, ...
    filternum, ...
    hsize, ...
    rate, ...
    momentum, ...
    iter, ...
    gpu_accel)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    
    %% reshape input data and labels
    data = cnnReshapeData(trainingData, gpu_accel);
    labels = reshapeLabels(labels, lsize, gpu_accel);
    
    [d,~,n] = size(data);
    convd = d-filtersize+1;
    k = convd/poolsize;
    
    %% initialise weights
    w = (rand(filtersize, filtersize, filternum)-0.5)*2;
    b = (rand(filternum, 1)-0.5)*2;
    w_i = (rand(k^2*filternum, hsize)-0.5)*2;
    w_o = (rand(hsize, lsize)-0.5)*2;
    
    dw_old = zeros(size(w));
    db_old = zeros(size(b));
    dw_i_old = zeros(size(w_i));
    dw_o_old = zeros(size(w_o));
    
    errors = zeros(1, iter);
    
                
    %% housekeeping

    dw = zeros(size(w));
    db = zeros(size(b));
    cnnConvDelta = zeros(convd, convd, filternum);

    % cast to gpuArray as necessary
    if gpu_accel
       dw = gpuArray(dw);
       db = gpuArray(db);
       cnnConvDelta = gpuArray(cnnConvDelta);
    end
    
    if gpu_accel
        dw_old = gpuArray(dw_old);
        db_old = gpuArray(db_old);
        dw_i_old = gpuArray(dw_i_old);
        dw_o_old = gpuArray(dw_o_old);
        w = gpuArray(w);
        b = gpuArray(b);
        w_i = gpuArray(w_i);
        w_o = gpuArray(w_o);
        errors = gpuArray(errors);
    end
    
    for e=1:iter
        errors(e) = 0;
        
        disp (['performing gradient descent batch ' num2str(e) ...
               ' of ' num2str(iter)]);
        
        
        %% perform stocastic gradient descent
        batch = randperm(n);
        
        for i=batch
            
            %tic;
            
            %% forward propagation
            cnnConvOutput = cnnConv(data(:,:,i), w, b, gpu_accel);
            cnnPoolOutput = cnnPool(cnnConvOutput, poolsize, gpu_accel);
            nnInput = cnnReshapePool(cnnPoolOutput, gpu_accel);
            nnHiddenOutput = nnLayer(w_i, nnInput);
            nnOutput = nnLayer(w_o, nnHiddenOutput);

            

            %% cost computation and back propagation
            nnOutputDelta = nnOutput.*(1-nnOutput).*(labels(i,:)-nnOutput);
            nnHiddenDelta = nnHiddenOutput .* ...
                            (1-nnHiddenOutput) .* ...
                            (nnOutputDelta * transpose(w_o));
            nnInputDelta = nnInput .* (1-nnInput) .* ...
                           (nnHiddenDelta * transpose(w_i));
            cnnPoolDelta = cnnUnrollPool(nnInputDelta, k, ...
                                         filternum, gpu_accel);
            
            for fn=1:filternum
                
                cnnConvDelta(:,:,fn) = (1/(poolsize^2))* ...
                    kron(cnnPoolDelta(:,:,fn), ones(poolsize));
                                             
            %% gradient computation
                dw(:,:,fn) = conv2(data(:,:,i), ...
                                   rot90(cnnConvDelta(:,:,fn),2), ...
                                   'valid');
                db(fn) = sum(sum(cnnConvDelta(:,:,fn)));
            end
            
            dw_o = transpose(nnHiddenOutput) * nnOutputDelta;
            dw_i = transpose(nnInput) * nnHiddenDelta;
            
            %% gradient descent
            w = w + rate * dw + momentum * dw_old;
            %b = b + rate * db + momentum * db_old;
            w_i = w_i + rate * dw_i + momentum * dw_i_old;
            w_o = w_o + rate * dw_o + momentum * dw_o_old;
            
            dw_old = dw;
            db_old = db;
            dw_i_old = dw_i;
            dw_o_old = dw_o;
            
            %% error computation
            errors(e) = errors(e) + sum(abs(labels(i,:)-nnOutput));
            
            %t = toc;
            
            %disp([' training sample ran in ' num2str(t) ' seconds.']);
        end
        
        disp (['L1 error: ' num2str(errors(e))]);
        
%         if e>1 && errors(e)>=errors(e-1)
%             disp('converged.');
%             break;
%         end
    end

    if gpu_accel
        errors = gather(errors);
    end
    
    plot(errors(1:e));
    
end

