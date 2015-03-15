function E = check_edges(E)
% Checks list of edges for invalid vertex ids
%
% Ryan A. Rossi, Purdue University
% Copyright 2012
%

if length(union(find(E(:,1) == 0), find(E(:,2) == 0)) > 0),
    fprintf('Error: A node with id 0 was found, adding 1 to all node ids \n')
    E(:,1) = E(:,1) + 1;
    E(:,2) = E(:,2) + 1;
end
