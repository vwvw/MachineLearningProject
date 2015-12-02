function [ output ] = cnnTrain(data, labels, k, w, w_i, w_o)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    [n, ~] = size(data);
    [m, ~, nconv] = size(w);
    
    
    for i=1:n
        
        img = single(rgb2gray(getImage(n)));
        
        for j=1:conv
            
            layer = conv2(img, w(:,:,j), 'valid');
    
    


end

