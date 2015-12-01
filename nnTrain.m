function [ w_i, w, w_o ] = nnTrain( data, label, lsize, hdim, hlayers, rate, momentum, iter, rng)
%NNTRAIN Summary of this function goes here
%   Detailed explanation goes here
    [n, dim] = size(data);
    
    % Initialise data
    w = (rand(hdim, hdim, hlayers)-0.5) * 2;
    w_i = (rand(dim, hdim)-0.5) * 2;
    w_o = (rand(hdim, lsize)-0.5) * 2;
    
    prev_w = zeros(hdim, hdim, hlayers);
    prev_w_i = zeros(dim, hdim);
    prev_w_o = zeros(hdim, lsize);
    
    errors = zeros(1, iter);
    prev_error = -1;
    for i=1:iter
        %do classification
            
        disp(['training (' num2str(i) ' out of ' num2str(iter) ')'])
        
        error = 0;
        for k=permute(rng, randperm(length(rng)))  
            %inst = randi([1 n]);
            inst = k;
            
            o_i = nnLayer(w_i, data(inst,:));
            
            
            o = zeros(hlayers, hdim);
            for j=1:hlayers
                if (j==1)
                    o(j,:) = nnLayer(w(:,:,j),o_i);
                else
                    o(j,:) = nnLayer(w(:,:,j),o(j-1,:));
                end
            end
            o_o = nnLayer(w_o,o(hlayers,:));
        
        
            %train data
            t = zeros(1,lsize);
            t(label(inst)+1) = 1;
        
            error = error + sum (abs(t-o_o));
            
            d_o = o_o.* (1-o_o).* (t - o_o);
            
            delta = zeros (hlayers, hdim);
            
            for j=hlayers:-1:1
                if (j==hlayers)
                    delta(j,:) = o(j,:) .* (1-o(j,:)) .* (d_o * transpose(w_o));
                else
                    delta(j,:) = o(j,:) .* (1-o(j,:)) .* (delta(j+1,:) * transpose(w(:,:,j+1)));
                end
            end
            
            d_i = o_i .* (1-o_i) .* (delta(1,:) * transpose(w(:,:,1)));
            
            
            dw_i = rate * transpose(data(inst,:)) * d_i;
            
            dw_o = rate * transpose(o(hlayers,:)) * d_o;
            
            dw = zeros(hdim,hdim,hlayers);
            
            for j=1:hlayers
                if(j==1)
                    dw(:,:,j) = rate * transpose(o_i) * delta(j,:);
                else
                    dw(:,:,j) = rate * transpose(o(j-1,:))*delta(j,:);
                end
            end
            w_i = w_i + dw_i + momentum * prev_w_i;
            w_o = w_o + dw_o + momentum * prev_w_o;
            w = w + dw + momentum * prev_w;
            
            prev_w = dw;
            prev_w_i = dw_i;
            prev_w_o = dw_o;
        end
         
        disp(['L1 error: ' num2str(error)]);
        
        errors(i) = error;
        if(prev_error<error && prev_error>0)
            disp(['converged.']);
            break;
        else
            prev_error = error;
        end
    end

    plot(errors(1:i));
    
end

