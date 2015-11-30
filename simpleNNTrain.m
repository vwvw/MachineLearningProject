function [ w_i, w_o ] = simpleNNTrain( data, label, lsize, hdim, rate, momentum, iter, rng)
%NNTRAIN Summary of this function goes here
%   Detailed explanation goes here
    [n, dim] = size(data);
    
    % Initialise data
    w_i = (rand(dim, hdim)-0.5)*20;
    w_o = (rand(hdim, lsize)-0.5)*20;
    dw_i_prev = zeros(dim, hdim);
    dw_o_prev = zeros(hdim, lsize);
    
    prev_error = -1;
    
    errors = zeros(iter);
    
    for i=1:iter
        %do classification
        disp(['training (' num2str(i) ' out of ' num2str(iter) ')'])
        
        error = 0;
        
        for k=permute(rng, randperm(length(rng)))
            %inst = randi([1 n]);
            inst = k;
            
            o_i = nnLayer(w_i, data(inst,:));
            o_o = nnLayer(w_o, o_i);
            
            %train data
            t = zeros(1,lsize);
            t(label(inst)+1) = 1;
            
            if(length(o_o)~=lsize || length(t)~=lsize)
                disp('dim mismatch');
                disp(o_o);
                disp('----');
                disp(t);
                disp(['dimension mismatch: ' num2str(length(o_o)) '; ' num2str(length(t))]);
            end
                 
            
            error = error + sum(abs(t-o_o));
            %{
            d_k = zeros(1, lsize);
            
            for l=1:lsize
               d_k(l) = o_k(l)*(1-o_k(l))*(t(l)-o_k(l)); 
            end
            
            d_h = zeros(1, hdim);
            for h=1:hdim
                summ = 0;
                for kk = 1:lsize
                   summ = summ + w_o(h,kk) * d_k(kk);
                end
                d_h(h) = o_h(h)*(1-o_h(h))*summ;
            end
            %}
            d_o = o_o.* (1-o_o).* (t - o_o);
            d_i = o_i .* (1-o_i) .* (d_o * transpose(w_o));
            %{
            dw_i = zeros(dim, hdim);
            dw_o = zeros(hdim, lsize);
            
            for h=1:hdim
                for kk=1:lsize
                    dw_o(h,kk) = rate * d_k(kk) * o_h(h);
                end
                
                for j=1:dim
                    dw_i(j,h) = rate * d_h(h)*data(inst,j);
                end
            end
            
            %}
                    
            
            dw_i = rate * transpose(data(inst,:)) * d_i;
            dw_o = rate * transpose(o_i) * d_o;
            
            w_i = w_i + dw_i + momentum * dw_i_prev;
            w_o = w_o + dw_o + momentum * dw_o_prev;
        end
        
        disp(['L1 error: ' num2str(error)]);
        
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
    
    plot(errors(1:i));
    
end

