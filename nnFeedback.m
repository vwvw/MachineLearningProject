function [ w, delta ] = nnFeedback( o, t, w_old, rate, x )
%NNFEEDBACK Summary of this function goes here
%   Detailed explanation goes here
    delta = o.* (1-o) .* t;
    w = w_old + rate * (delta .* x);

end

