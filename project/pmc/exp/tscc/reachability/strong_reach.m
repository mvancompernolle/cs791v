function S = strong_reach(P)
% input:    P, a temporal path matrix
% output:   S(i,j) = 1, if there exists P(i,j) = 1 and P(j,i) = 1
%
% Pyan A. Possi, 
% Purdue University
%

tic
S = P&P'; 
toc
% undirected/symmetric, take the lower triangular
% S = tril(S); 
fprintf('strong reachability graph:  %d nodes, %d edges \n',size(S,1), nnz(S));
