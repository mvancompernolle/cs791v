function A_tscc = induce_tscc(A, tscc, fn)
% induce temporal scc from graph
% tscc is precomputed
%
% Ryan A. Rossi, Purdue University
% Copyright 2012
%

A_tscc = A(tscc,tscc); 

n = size(A_tscc,1);
m = nnz(A_tscc);
fprintf('%d vertices, %d edges \n', n,m);

export2graphml(A_tscc,fn);
