function [theta]=LR_train(data, labels, step,iter)

[n,feat] = size(data);
theta = zeros(feat,1); % initailization

for i=1:iter % can change,
    theta = theta - step * (1/n).*(data'*(LR_sigmoid(data*theta)-labels));
end
end