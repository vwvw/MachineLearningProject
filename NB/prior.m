function [p] = prior(yTrain)
yTrain= yTrain+1;
numberOfElement = size(yTrain,1);
uniqueElementsMatrix = unique(yTrain)
numberOfUniqueElements = size(uniqueElementsMatrix,1);
p = zeros(numberOfUniqueElements,1);

for i = transpose(uniqueElementsMatrix)
    incr = 0;
    for d = 1:numberOfElement
        if i == yTrain(d,1)
            incr= incr+1;     
        end
    end
    p(i)=incr/numberOfElement;
end
end
