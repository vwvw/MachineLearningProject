function [ w_i, w, w_o ] = nnTrainMultiple( data, labels, lsize, hdims, hlayers, rates, iters )
%NNTRAINMULTIPLE Summary of this function goes here
%   Detailed explanation goes here
    
    n = length(hdims);
    disp(n)
    w_i(1,n) = gpuArray(1);
    w(1,n) = gpuArray(1);
    w_o(1,n) = gpuArray(1);


    parfor i=1:n
        [w_i(i), w(i), w_o(i)] = nnTrain(data, labels, lsize, hdims(i), hlayers(i), rates(i), iters(i));
    end
    
end

