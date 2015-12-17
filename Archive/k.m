function [ cor ] = k(data, labels, Model)  
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
cor = zeros(50,1);
parfor k = 40 : 75
    disp(k)
    training_tree = Model.tree;
   output =  knnClassify( data, training_tree, k);
    cor (k)= correctness(output, labels);
end

