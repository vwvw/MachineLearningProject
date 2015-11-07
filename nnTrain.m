function [ w_i, w, w_o ] = nnTrain( input, data, label, lsize, hdim, hlayers, rate, iter)
%NNTRAIN Summary of this function goes here
%   Detailed explanation goes here
    [n, dim] = size(data);
    [~, ldim] = size(label);
    % Initialise data
    w = rand(hdim, hdim, hlayers);
    w_i = rand(hdim, dim);
    w_o = rand(ldim, hdim);
    
    for i=1:iter
        for k=1:n
            %do classification
            o_i = nnLayer(w_i, data(k,:));
        
            o = zeros(hlayers, hdim);
            for j=1:hlayers
                if (j==1)
                    o(j,:) = nnLayer(w(:,:,j),o_i);
                else
                    o(j,:) = nnLayer(w(:,:,j),o(j-1,:));
                end
            end
            o_o = nnLayer(w_o,o(:,:,hlayers));
        
        
            %train data
            t = zeros(1,lsize);
            t(labels(k)+1) = 1;
            
            d_o = o_o.* (1-o_o).* (t - o_o)
            
            delta = zeros (hlayers, hdim);
            
            for j=hlayers:-1:1
                if (j==hlayers)
                    delta(j,:) = o(j,:) .* (1-o(j,:)) .* sum(transpose(w_o) .* d_o);
                else
                    delta(j,:) = o(j,:) .* (1-o(j,:)) .* sum(transpose(w(:,:,j+1) .*delta(j+1,:)));
                end
            end
            delta
            
            d_i = o_i .* (1-o_i) .* sum(transpose(w(:,:,1) .* delta(1,:)))
        end
    end

end

