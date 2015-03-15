function [ d ] = is_directed( A )

d = 1;
if all(all(tril(A) == A)),
    fprintf('lower triangular, undirected \n \n');
    %A = A|A'; todo
    d = 0;
elseif all(all(A==A')),
    fprintf('symmetric, undirected \n')
    d = 0;
end




