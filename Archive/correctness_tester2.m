function [correctness, output] = correctness_tester2(data, labels, w_in, w, w_out)
    [n,~] = size(data);
    
    correctness = 0;
    
    out = nnClassify(data, w_in, w, w_out);
    
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