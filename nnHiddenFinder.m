function [tr, te] = nnHiddenFinder(feat_train, labels_train, feat_test, labels_test)

    trainingAccuracy = zeros(300, 1);
    testAccuracy = zeros(300, 1);

    rate = 0.0001;
    momentum = 0.8;
    sampleRange = 1:5000;

    parfor i = 600:899
        disp(['nHidden=' num2str(floor(i/3)) ' iter=' num2str(mod(i,3)) ' started']);
        
        [a,b] = simpleNNTrain(feat_train, labels_train, 10, floor(i/3), rate, momentum, 1000, sampleRange);
        
        [testAccuracy(i-599),~] = correctness_tester(feat_test, labels_test, a, b);
        [trainingAccuracy(i-599),~] = correctness_tester(feat_train, labels_train, a, b);
        disp(['nHidden=' num2str(floor(i/3)) ' iter=' num2str(mod(i,3)) ' finished']);
        
    end
    
    tr = reshape(trainingAccuracy, 3, 100);
    te = reshape(testAccuracy, 3, 100);
end

