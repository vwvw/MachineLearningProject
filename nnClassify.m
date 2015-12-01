function [ class ] = nnClassify( data, w_in, w, w_out )
%NNCLASSIFY Summary of this function goes here
%   Detailed explanation goes here
    [n, dim] = size(data);
    
    [~,~,nhlayers] = size(w);
    
    out = nnLayer(w_in, data);
    
    out_gathered = gather(out)*255;
    disp([min(data), max(data)]);
    % Create a 32x32 color image.
    %image = zeros(32,32,3, 'uint8');
    %image(:,:,1)=reshape(out_gathered,32,32);
    %image(:,:,2)=reshape(out_gathered,32,32);
    %image(:,:,3)=reshape(out_gathered,32,32);
    %Rotate it in the corect direction
    %image = rot90(image,3);

    %subplot(nhlayers+1,1, 1), subimage(image);
    for i=1:nhlayers
        out = nnLayer(w(:,:,i), out);
        out_gathered = gather(out) * 255;
        %disp([min(out_gathered), max(out_gathered)]);
        
        % Create a 32x32 color image.
        %image = zeros(32,32,3, 'uint8');
        %image(:,:,1)=reshape(out_gathered,32,32);
        %image(:,:,2)=reshape(out_gathered,32,32);
        %image(:,:,3)=reshape(out_gathered,32,32);
        %Rotate it in the corect direction
        %image = rot90(image,3);
        
        %subplot(nhlayers+1,1, (i+1)), subimage(image);
        
    end
    
    class = nnLayer(w_out, out);
end

