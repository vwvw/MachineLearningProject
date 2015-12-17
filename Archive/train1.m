function [ Model ] = train1( X, Y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

Model = struct('theta',LR_train_multiple(kMeansPrep(X)', Y, 0.1,3000));
end

