function [ output ] = classify2( Model, X )
%Using the model trained in train.m to classify the image of X

%model contain w_in in the first column and v_out in the second column
%X is the test data
%output is a vector of the length of X, contianing the class wich was
%attridbuted to the image.

    b_i = Model{1};
    b_o = Model{2};
    w_i = Model{3};
    w_o = Model{4};
    
    [n,~] = size(X);
    
    feat = features(X);
    
    out = simpleNNClassify(feat, w_i, w_o, repmat(b_i, n, 1), repmat(b_o, n, 1));
    
    output = zeros(n,1);
    
    for i = 1:n
        [~, lbl] = max(out(i,:));
        output(i) = lbl-1;
    end

end

