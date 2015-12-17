function [correctness, output] = correctness_tester(data, labels, w_in, w_out, b_i, b_o)
    [n,~] = size(data);
    
    correctness = 0;
    
    out = simpleNNClassify(data, w_in, w_out, repmat(b_i, n, 1), repmat(b_o, n, 1));
    
    output = zeros(n,1);
    
    for i = 1:n
        [~, lbl] = max(out(i,:));
        output(i) = lbl-1;
        if(output(i)==labels(i))
            correctness = correctness + 1;
        end
    end
    
    correctness = correctness / n;
end