function [t] = naiveBayesClassify(xTest, M, V, p)
    M = transpose(M);
    V = transpose(sqrt(V));
    len = length(p);
    
    [N,~] = size(xTest);
    
%     t = arrayfun( ...
%           @(x) ...
%             maxId( ...
%               arrayfun( ...
%                 @(y) prod(normpdf(xTest(x,:),M(y,:),V(y,:)))*p(y), ...
%                 1:len)), ...
%           1:length(xTest));
%     t = transpose(t);
    

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
