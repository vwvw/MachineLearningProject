function [ model ] = train( X,Y )
%training our Model
[N, D] = size(X);
feat  = features(single( X) );
save('feat.mat', 'feat');
[ w_in, w_out ] = simpleNNTrain( feat, Y, 10, 17, 0.1, 0.1, 1000, 1:N/2);
field = 'f';
value = {w_in; w_out};
model = struct(field, value);
end

