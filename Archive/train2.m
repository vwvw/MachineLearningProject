function [ Model ] = train2(data, label)

    feat = features(data);
    
    [w_i, w_o, b_i, b_o] = simpleNNTrain(feat, label, 10, 236, 0.001, 0.8, 1000, 1:5000);
    
    Model = {b_i, b_o, w_i, w_o};

end