function [ BPQ ] = knn_find_k_closest( ref_point, tree, BPQ,d )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
root_dist = knn_dist((ref_point)', double(tree.node(1:end-1)));
BPQ = BPQ_add(BPQ, root_dist, tree.node(1:end));
if ~isstruct(tree.right_tree)
   if ~isstruct(tree.left_tree)
        % do nothing
   else
        BPQ = BPQ_add(BPQ, knn_dist(ref_point, tree.left_tree.node), tree.left_tree.node);
   end 
else
    if BPQ(end,1)>abs(ref_point(1,d)-tree.node(d,1))% cut the line
        BPQ =  knn_find_k_closest(ref_point, tree.right_tree, BPQ,d+1);
        BPQ = knn_find_k_closest(ref_point, tree.left_tree, BPQ,d+1);
    else
        if ref_point(1,d)<tree.node(d,1) %% left of the cutting line
           BPQ= knn_find_k_closest(ref_point, tree.left_tree, BPQ,d+1);
        else
           BPQ = knn_find_k_closest(ref_point, tree.right_tree, BPQ,d+1);
        end

    end

end

end

