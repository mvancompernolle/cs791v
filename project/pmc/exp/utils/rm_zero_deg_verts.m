function [ A remove_idx ] = rm_zero_deg_verts( A )
% remove verts with zero deg [i.e., sum(A(i,:)) = 0 and sum(A(:,i))=0]
% useful to use after A&A' transformation
% 
% Ryan A. Rossi, Purdue University
% Copyright 2012
%

zero_indeg = find(sum(A) == 0);
zero_outdeg = find(sum(A') == 0);
remove_idx = intersect(zero_indeg, zero_outdeg);
if ~isempty(remove_idx),
    A(remove_idx,:) = [];
    A(:,remove_idx) = [];
end
fprintf('%d vertices with zero degree removed \n', length(remove_idx));

