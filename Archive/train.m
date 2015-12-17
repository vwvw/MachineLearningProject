function [ Model ] = train( X,Y )
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here
[w, b] = nnTrain(features(X), Y, 10, 3, 1, 0.1, 0.8, 1000, 1:5000);
Model = {w,b};

end

