[w_in, w_out]=simpleNNTrain(feat, labels, 10, 10, 0.001, 0.1, 100, [1:5000]);
[correctness, output] = correctness_tester(feat, labels, w_in, w_out);
correctness