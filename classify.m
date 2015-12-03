function [ output ] = classify( Model, X )
%Using the model trained in train.m to classify the image of X

%model contain w_in in the first column and v_out in the second column
%X is the test data
%output is a vector of the length of X, contianing the class wich was
%attridbuted to the image.

    k = 5;
    km = Model{1};
    labels = Model{2};
    
    tic;
    output = knnClassify(X, km, k, labels);

    dt = toc;
    
    disp (['used ' num2str(dt) ' seconds.']);

end

