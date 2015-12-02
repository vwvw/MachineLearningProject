function [M, V] = likelihood(xTrain, yTrain)
    Y = double(unique(yTrain));
    C = length(xTrain(1,:));
    D = horzcat (xTrain, double(yTrain));


    M = arrayfun (@(y) ...
                    arrayfun (@(c) mean(D(D(:,end)==y,c)), 1:C), ...
                  Y,'UniformOutput',false);
    V = arrayfun (@(y) ...
                    arrayfun (@(c) var(D(D(:,end)==y,c)), 1:C), ...
                  Y,'UniformOutput',false);

    M = transpose(cell2mat(M));
    V = transpose(cell2mat(V));
end
