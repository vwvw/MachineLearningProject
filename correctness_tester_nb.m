function [correctness, output] = correctness_tester_nb(data, labels, M,V,p)
    [n,~] = size(data);
    
    correctness = 0;
    
    output = uint8(naiveBayesClassify(data, M, V, p));
    
    for i = 1:n
        if(output(i)==labels(i))
            correctness = correctness + 1;
        end
    end
    
    correctness = correctness / n;
end