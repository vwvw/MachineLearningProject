function [ out ] = sigmoid( y, x, w )
%SIGMOID Sigmoid activation function.

    out = 1./(1+exp(-(x*w)));
    
    if y
        out = 1-out;
    end
end

