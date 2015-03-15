function [ A ] = convert_adj( edges )


if ~isempty(find(edges(:,1:2) == 0)),
    fprintf('ids must be positive... incrementing ids by 1 \n');
    edges(:,1:2) = edges(:,1:2) + 1;
end

A = spconvert([edges(:,1:2) ones(size(edges,1),1)]);
A = spones(A); %Replace nonzero sparse matrix elements with ones

[n m] = size(A);
if n ~= m,
   A(max([n,m]), max(n,m)) = 0; 
end

