function [t] = naiveBayesClassify(xTest, M, V, p)
    [nbrX, nbrClass]= size(M);
    nbrTry = size(xTest,1);
    t = zeros( nbrTry,1);
    for triesNumber =  1 : nbrTry
        classifier = zeros(1,nbrClass);
        for class = 1:nbrClass
            probaX = zeros(1,nbrX);
            for xVar = 1: nbrX
                probaX(xVar) = normpdf(xTest(triesNumber,xVar),M(xVar, class),sqrt(V(xVar, class)));
            end
            classifier(class) = prod(probaX)*p(class);

        end
        t(triesNumber,1) = find(classifier==max(classifier),1);
    end
end
