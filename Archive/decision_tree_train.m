function [ tree ] = decision_tree_train( data, labels, level)

[n,feat] = size(data);
if n > 20 
    info = mutual_info(data,labels);
    [max_info,dim] = max(info);
    if max_info > 0.1 
        m = mean(data(:,dim));
        left_data = data(data(:,dim)<m,:);
        right_data = data(data(:,dim)>=m,:);
        left_labels = labels(data(:,dim)<m,1);
        right_labels = labels(data(:,dim)>=m,1);

        left_tree = decision_tree_train(left_data, left_labels, level-1);
        right_tree = decision_tree_train(right_data, right_labels, level-1);

        tree = struct('mean', m,'dim', dim, ...
          'right_tree', right_tree, 'left_tree', left_tree); % storing the dim and the value to cut at
    else
       tree = mode(labels); 
    end
    
else
    tree = mode(labels);
end


end
