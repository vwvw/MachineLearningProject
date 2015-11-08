function [ w_i, w, w_o ] = nnTrain( data, label, lsize, hdim, hlayers, rate, iter, rng)
%NNTRAIN Summary of this function goes here
%   Detailed explanation goes here
    [n, dim] = size(data);
    data = horzcat (repmat (1,n,1), data);
    [n, dim] = size(data);
    
    % Initialise data
    w = gpuArray(rand(hdim, hdim, hlayers));
    w_i = gpuArray(rand(dim, hdim));
    w_o = gpuArray(rand(hdim, lsize));
    
    for i=1:iter
            %do classification
            
            disp(['training (' num2str(i) ' out of ' num2str(iter) ')'])
         for k=rng  
            %inst = randi([1 n]);
            inst = k;
            
            o_i = nnLayer(w_i, data(inst,:));
            
            
            o = gpuArray(zeros(hlayers, hdim));
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
            t = gpuArray(t);
            
            d_o = o_o.* (1-o_o).* (t - o_o);
            
            delta = gpuArray(zeros (hlayers, hdim));
            
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
            
            dw = gpuArray(zeros(hdim,hdim,hlayers));
            
            for j=1:hlayers
                if(j==1)
                    dw(:,:,j) = rate * transpose(o_i) * delta(j,:);
                else
                    dw(:,:,j) = rate * transpose(o(j-1,:))*delta(j,:);
                end
            end
            w_i = w_i + dw_i;
            w_o = w_o + dw_o;
            w = w + dw;
         end
    end

end

