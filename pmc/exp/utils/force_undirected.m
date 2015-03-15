function [ output_args ] = force_undirected( A, opts )
% 
% Force A to be undirected, since some problems require undirected graphs
% Choice of how to transform the graph, depends on the graph itself and
% the application.
% 
% Two options:
% ============================================
% if opts == 1, then AND adjacency matrix
%     edge exists iff A(i,j)=1 AND   A(j,i)=1
%
% if opts == 2, then OR adjacency matrix
%     edge exists if A(i,j)=1  OR if A(j,i)=1
%
%
% Ryan A. Rossi, Purdue University
% Copyright 2012


if nargin < 2,
    opts = 1; 
end

if opts < 1 || opts > 2,
    error('opts must be 1 for A&A^T or 2 for A|A^T');
end

if opts == 1,
    A = A&A';
elseif opts == 2,
    A = A|A';
end


end

