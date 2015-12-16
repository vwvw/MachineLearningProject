function [ class ] = simpleNNClassify( data, w_in, w_out, b_i, b_o )
%NNCLASSIFY Summary of this function goes here
%   Detailed explanation goes here
    out = nnLayer(w_in, data, b_i);
    
    class = nnLayer(w_out, out, b_o);
end