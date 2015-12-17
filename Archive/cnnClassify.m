function [out] = cnnClassify (data, labels, W, b, poolsize, w_i, w_o, gpu_accel)
%cnnClassify convolutional neural network classifier

    %% reshape input data
    data = cnnReshapeData(data, gpu_accel);

    %% convolutional layer
    convd = cnnConv(data, W, b, gpu_accel);
    
    %% pooling layer
    poold = cnnPool(convd, poolsize, gpu_accel);
    
    %% reshape pooled data
    reshaped = cnnReshapePool(poold, gpu_accel);
    
    %% fully-connected final layer
    out = simpleNNClassify(reshaped, w_i, w_o);
end