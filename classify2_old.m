function [ output ] = classify2( Model2, X )
%Using the model trained in train.m to classify the image of X

%model contain w_in in the first column and v_out in the second column
%X is the test data
%output is a vector of the length of X, contianing the class wich was
%attridbuted to the image.
    b_i = Model2(1).val;
    b_o = Model2(2).val;
    w_i = Model2(3).val;%val = name of the field in the struct
    w_o = Model2(4).val;
    
    [n,~] = size(X);
    
    feat = features(X);
    
    out = simpleNNClassify(feat, w_i, w_o, repmat(b_i, n, 1), repmat(b_o, n, 1));
    
    output = zeros(n,1);
    
    for i = 1:n
        [~, lbl] = max(out(i,:));
        output(i) = lbl-1;
    end

end

