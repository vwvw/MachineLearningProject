function [p] = sigmoidProb(y, x, w)
    val = 1./(1+ exp(dot(x,w)));
    if(y== 0)
        p = val;
    else p = 1-val;
    end

end
