function [ output ] = classify3( Model, X )

    %classify with deep neural network
    w = Model{1};
    b = Model{2};
    
    output = nnClassify(features(X), w, b);

end

