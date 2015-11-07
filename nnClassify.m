function [ class ] = nnClassify( data, w_in, w, w_out )
%NNCLASSIFY Summary of this function goes here
%   Detailed explanation goes here
    [n, dim] = size(data);
    
    [~,~,nhlayers] = size(w);
    
    out = nnLayer(w_in, data);
    
    for i=1:nhlayers
        out = nnLayer(w(:,:,i), out);
    end
    
    class = nnLayer(w_out, out);
end

