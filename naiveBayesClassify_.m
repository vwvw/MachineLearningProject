function [t] = naiveBayesClassify(xTest, M, V, p)
    len = length(p);
    
    [N,~] = size(xTest);

    t = zeros(N, 1);
    
    lp = log(p);
    
    for i=1:N
        classes = arrayfun (@(y) sum(log(normpdf(xTest(i,:),M(y,:),V(y,:)))) + lp(y), 1:len);
        
        t(i) = maxId(classes)-1;
    end
    
end


function [v] = maxId(arr)
    [~,v] = max(arr);
end
