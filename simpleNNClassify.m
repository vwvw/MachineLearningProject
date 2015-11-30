function [ class ] = simpleNNClassify( data, w_in, w_out )
%NNCLASSIFY Summary of this function goes here
%   Detailed explanation goes here
    out = nnLayer(w_in, data);
    
    class = nnLayer(w_out, out);
end