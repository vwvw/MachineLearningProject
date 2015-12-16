function [ v_out ] = nnLayer( w, v_in, b )
%NNLAYER Summary of this function goes here
%   Detailed explanation goes here
    v_out = 1./ (1 + exp (- (v_in * w + b)));
end