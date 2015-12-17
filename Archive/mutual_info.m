function [ gain ] = mutual_info( data, labels)

[n,f] = size(data);

l = zeros(10,1);
for i = 1 :n
    l(labels(i)+1) =l(labels(i)+1)+1; 
end
entropyLabels = sum(l./sum(l).*log2(l./sum(l)));

entropy = zeros(f,1);
for feat = 1 : f
    d = data(:,f);
    mi = min(d);
    occurences = zeros(max(d)-mi+1,1);
    for i = 1 : n
       occurences(d(i)-mi+1) = occurences(d(i)-mi+1)+1;
    end
    p = occurences./sum(occurences);
    entropy(feat,1) = -sum(p(:).*log2(p(:)),'omitnan');    
end

gain = zeros(f,1);
for feat = 1:f
   gain(feat,1) = entropyLabels-entropy(feat,1);
end


end

