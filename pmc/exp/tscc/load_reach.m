function [S]=load_reachability(graphname)
% LOAD_GRAPH Loads a graph from the data directory
%
% load_graph is a helper function to load a graph provided with the
% regardless of the current working directory.  
%
% Example:
%   A = load_graph('twitter-reach');

% David F. Gleich and Ryan Rossi
% Copyright 2012, Purdue University

path=fileparts(mfilename('fullpath'));
load(fullfile(path,'graph-reachability-db',[graphname '.mat']));
S=spones(S); 
S = S - diag(diag(S));