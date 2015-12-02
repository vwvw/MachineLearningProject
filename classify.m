function [ correctness ] = classify( Model, X,feat )
%Using the model trained in train.m to classify the image of X

%model contain w_in in the first column and v_out in the second column
%X is the test data
%output is a vector of the length of X, contianing the class wich was
%attridbuted to the image.

[correctness, output] = correctness_tester(feat, zeros(size(X,1)), Model(1).f, Model(2).f)

end

