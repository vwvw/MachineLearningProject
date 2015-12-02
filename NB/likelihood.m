function [M, V] = likelihood(xTrain, yTrain)

    yProbability = prior(yTrain);
    nbrClass = size(yProbability,1);
    nbrXVariable = size(xTrain,2);
    nbrTries = size(xTrain,1);
    M=zeros(nbrXVariable,nbrClass);
    V=zeros(nbrXVariable,nbrClass);
    for c = 1: nbrClass
        incr = 0;
        meansForC = zeros(nbrXVariable,1);
        for tryNbr = 1: nbrTries
            if yTrain(tryNbr) == c %good class
                incr = incr+1;
                meansForC = meansForC+transpose(xTrain(tryNbr,:));
            end
        end
        M(:,c) = meansForC./incr;
    end

    for c = 1: nbrClass
        incr = 0;
        total = zeros(nbrXVariable,1);
        for tryNbr = 1: nbrTries
            if yTrain(tryNbr) == c
                incr = incr+1;
                total = total+(transpose(xTrain(tryNbr,:))-M(:,c)).^2; %variance calculations
            end
        end
        V(:,c) = total/incr;
    end
end