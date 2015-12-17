function [ theta ] = LR_train_multiple( data, labels, step, iter )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

c = 10;
[n,feat] = size(data);
theta = zeros(c,feat);
for class = 1 : c
    
    theta(class,:)=LR_train(data, labels==class, step,iter);
    
    
end

end

