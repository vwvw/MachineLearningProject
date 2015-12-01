function [dc  ] = dominantColor( data,i )

h = fspecial('average',3);%bluring
A = getImage(data, i);
blur = imfilter(A, h, 'replicate');
%figure
%image(blur)
dc = [0,0,0];
dominantColorOccurences = -Inf;
A = double(A);
for i = 1 : 32
    for j = 1 : 32
        if mod(i+j,2) == 0
           R = A(i,j,1);
           G = A(i,j,2);
           B = A(i,j,3);
           if [R,G,B] ~= dc 
               counter = 0;
               for k = 1 : 32
                   for l = 1 : 32
                        if abs(R- A(k,l,1))+abs(G- A(k,l,2))+abs(B- A(k,l,3)) < 20
                            counter= counter +1;
                        end
                   end
               end
               if(counter > dominantColorOccurences)
                   dominantColorOccurences = counter;
                   dc = [R,G,B];
               end
           end
        end
    end
end
% result = zeros(1,1,3,'uint8');
% for k = 1 : 3
%   result(1,1,k) = dominantColor(1,k);
% end
% figure
% image(result)
end

