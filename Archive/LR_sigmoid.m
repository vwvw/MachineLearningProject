function [ output_args ] = LR_sigmoid( input )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
output_args = 1.0 ./(1.0+exp((-1).*input));

end

