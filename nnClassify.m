function [ class ] = nnClassify( data, w, b )
%NNCLASSIFY Summary of this function goes here
%   Detailed explanation goes here
    [n, dim] = size(data);
    
    [~,numLayers] = size(b);
    [~,numTransitions] = size(w);
    

    output = cell(1, numLayers);
            
    % forward
    output{1} = data ;%+ b{1};
    for l=1:numTransitions
       output{l+1} = nnLayer(w{l},output{l}, repmat(b{l+1},  n, 1));
    end
    
    [~, class] = max(output{numLayers}, [], 2);
    class = class - 1;
    
end

