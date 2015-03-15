function A=load_graph(graphname)
% LOAD_GRAPH Loads a graph from the data directory
%
% Updated for max cliques/tscc (Ryan Rossi)
%   - for MC, consider only the largest component
%   - if directed, remove nonreciprocal edges, (hence, if undirected, 
%       then must be symmetric)
%   
%
% load_graph is a helper function to load a graph provided with the
% regardless of the current working directory.  
%
% Example:
%   A = load_graph('retweet-obama');

% David F. Gleich and Ryan Rossi
% Copyright 2012, Purdue University


path=fileparts(mfilename('fullpath'));
load(fullfile(path,'graph-db',[graphname '.mat']));
A = A&A';                   % remove nonreciprocal edges
A=spones(A);                % weights discarded
A = largest_component(A);   % largest SCC
A = A - diag(diag(A));      % discard self-loops
