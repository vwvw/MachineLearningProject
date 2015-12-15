function [ BPQ ] = BPQ_add( BPQ, dist, point )
%Bounded priority Queue, biggest last
[k,dim] = size(BPQ);
%fist component is dist to p;
val = k+1;

while  val >1 && dist < BPQ(val-1,1)
        val= val-1;
end
if k+1 ~= val
    BPQ(val+1:end,:) = BPQ(val:end-1,:);
    BPQ(val,:) = [dist, point'];    
end



end

