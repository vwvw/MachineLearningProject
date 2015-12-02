function [p] = prior(yTrain)
    Y = unique(yTrain);
    len = length(yTrain);
    p = arrayfun (@(y) length(yTrain(yTrain==y))/len, Y);
end
