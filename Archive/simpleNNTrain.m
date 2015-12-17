function [ w_i, w_o, b_i, b_o ] = simpleNNTrain( data, labels, lsize, hdim, rate, momentum, iter, rng)
%NNTRAIN Summary of this function goes here
%   Detailed explanation goes here

    [~, dim] = size(data);
    
    % Initialise data
    
    % with Xavier initialisation
    w_i = normrnd(0, sqrt(2/(dim)), dim, hdim);
    w_o = normrnd(0, sqrt(2/(hdim)), hdim, lsize);
    b_i = normrnd(0, sqrt(2/(dim)), 1, hdim);
    b_o = normrnd(0, sqrt(2/(hdim)), 1, lsize);
    dw_i_prev = (zeros(dim, hdim));
    dw_o_prev = (zeros(hdim, lsize));
    db_i_prev = zeros(1, hdim);
    db_o_prev = zeros(1, lsize);
    
    prev_error = -1;
    
    errors = zeros(1, iter);
    
    for i=1:iter
        %do classification
        disp(['training (' num2str(i) ' out of ' num2str(iter) ')'])
        
        error = 0;
        
        for k=permute(rng, randperm(length(rng)))
            %inst = randi([1 n]);
            inst = k;
            
            
            o_i = nnLayer(w_i, data(inst,:), b_i);
            o_o = nnLayer(w_o, o_i, b_o);
            
            %train data
            t = zeros(1,lsize);
            t(labels(inst)+1) = 1;
            
            if(length(o_o)~=lsize || length(t)~=lsize)
                disp('dim mismatch');
                disp(inst);
                disp('----');
                disp(t);
                disp(['dimension mismatch: ' num2str(length(o_o)) '; ' num2str(length(t))]);
                continue;
            end
                 
            error = error + sum((t-o_o).^2);

            
            d_o = o_o.* (1-o_o).* (t - o_o);
            d_i = o_i .* (1-o_i) .* (d_o * transpose(w_o));

            
            dw_i = transpose(data(inst,:)) * d_i;
            dw_o = transpose(o_i) * d_o;
            db_o = d_o;
            db_i = d_i;
            
            
            w_i = w_i + rate *dw_i + momentum * dw_i_prev;
            w_o = w_o + rate *dw_o + momentum * dw_o_prev;
            b_i = b_i + rate *db_i + momentum * db_i_prev;
            b_o = b_o + rate *db_o + momentum * db_o_prev;
            
            
            dw_i_prev = dw_i;
            dw_o_prev = dw_o;
            db_i_prev = db_i;
            db_o_prev = db_o;
        end
        
        disp(['L1 error: ' num2str(error./length(rng))]);
        
        errors(i) = error;
        
        if(prev_error <= error && prev_error ~= -1)
            disp('converged.');
           break;
        else
            prev_error = error;
            dw_i_prev = dw_i;
            dw_o_prev = dw_o;
        end
    end
    % errors = gather(errors);
    disp (['error: ' num2str(errors(i))]);
    plot(errors(1:i));
    
end

