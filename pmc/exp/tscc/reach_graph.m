function [S R] = reach_graph(E, fn, is_und)
%========================================================
% Compute (strong) reachability graph
%
% The strong reach graph, S, is used for computing the largest temporal
% strong component (TSCC). The other graph R might be useful for analysis
%
% Note that these matrices may become extremely dense
%
%
% INPUT:
%           E:     list of edges w/ times, (i, j, time)
%
% OUTPUT:
%           R:     n x n adj matrix, directed graph
%                  R(i,j)=1 if i can temporally reach j
%                  (if there exists at least a single
%                       temporal path between i and j)
%
%           S:     n x n adj matrix, undirected graph
%                  S(i,j)=1 if R(i,j)=1 AND R(j,i)=1
%
%           cc_fname.mtx:
%               file with reachability edges
%
%           scc_fname.mtx:
%               file with strong reachability edges for temporal SCC

% Ryan A. Rossi, Purdue University
% Copyright 2012

if nargin < 1, error('edge list undefined.'); end;
if nargin < 2, fn = 'untitled'; end;
if nargin < 3, is_und = 0; end;

dbpath = 'graph-reachability-db/';
[~,name] = fileparts(fn);

if is_und,
    R = reach(E, is_und);
else
    R = reach(E);
end


if nargout > 1,
    Rcc = R|R';
    save([dbpath,name,'_cc'],'Rcc','-v7.3');
    write_mtx(Rcc,['cc_',fn]);
    clear Rcc
end

S = strong_reach(R);

tic; write_mtx(S,['scc_',fn]); toc;

save([dbpath,name,'_scc'],'S','-v7.3');
save([dbpath,name,'_R'],'R','-v7.3');
