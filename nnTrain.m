function [w, b] = nnTrain( data, labels, lsize, numHiddenLayers, ratei, rated, momentum, iter, rng)
%NNTRAIN Summary of this function goes here
%   Detailed explanation goes here
    [n, dim] = size(data);
    
    numLayers = numHiddenLayers + 2;
    numTransitions = numLayers - 1;
    
    interpolate = (dim - lsize)/numTransitions;
    
    numNeurons = zeros(1, numLayers);
    numNeurons(1) = dim;
    numNeurons(numLayers) = lsize;
    
    for i=1:numHiddenLayers
        numNeurons(i+1) = floor(lsize + interpolate * (numLayers-i-1));
    end
    
    numNeurons
    
    % initialise values
    w = cell(1, numTransitions);
    b = cell(1, numLayers);
    dw_prev = cell(1, numTransitions);
    db_prev = cell(1, numLayers);
    
    for i=1:numTransitions
        w{i} = normrnd(0, sqrt(2/(numNeurons(i))), numNeurons(i), numNeurons(i+1));
        dw_prev{i} = zeros(numNeurons(i), numNeurons(i+1));
    end
    
    for i=1:numLayers
        b{i} = normrnd(0, sqrt(2/numNeurons(i)), 1, numNeurons(i));
        db_prev{i} = zeros(1, numNeurons(i));
    end
    
    errors = zeros(1, iter);
    num = 0;
    % begin training
    for i=1:iter
        %do classification
        disp(['training (' num2str(i) ' out of ' num2str(iter) ')'])
        
        
        for k=permute(rng, randperm(length(rng)))
            
            rate = ratei/(1+num*rated);
            num = num+1;
            
            output = cell(1, numLayers);
            
            % forward
            output{1} = data(k, :) ;%+ b{1};
            for l=1:numTransitions
               output{l+1} = nnLayer(w{l},output{l}, b{l+1});
            end
            
            % expected output
            t = zeros(1,lsize);
            t(labels(k)+1) = 1;
            
            if(length(output{numLayers})~=lsize || length(t)~=lsize)
                disp('dim mismatch');
                disp(k);
                disp('----');
                disp(t);
                disp(['dimension mismatch: ' num2str(length(o_o)) '; ' num2str(length(t))]);
                continue;
            end
            
            errors(i) = errors(i) + sum((t-output{numLayers}).^2);
            
            delta = cell(1, numLayers);
            
            delta{numLayers} = output{numLayers} .* (1-output{numLayers}).* (t - output{numLayers});
            
            for l=numTransitions:-1:1
                delta{l} = output{l} .* (1-output{l}) .* (delta{l+1} * transpose(w{l}));
            end
            
            dw = cell(1, numTransitions);
            
            for l=1:numTransitions
               dw{l} = transpose(output{l}) * delta{l+1};
               w{l} = w{l} + rate * dw{l} + momentum * dw_prev{l};
            end
            
            
            for l=1:numLayers
               b{l} = b{l} + rate * delta{l} + momentum * db_prev{l};
            end
            
            db_prev = delta;
            dw_prev = dw;
        end
        
        disp(['L2 error: ' num2str(errors(i)/length(rng))]);
        
        if((i>1) && (errors(i-1) <= errors(i)))
           disp('converged.');
           break;
        end
        
    end
    
    
    disp (['error: ' num2str(errors(i))]);
    plot(errors(1:i));
    
end

