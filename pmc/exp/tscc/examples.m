% set appropriate paths:
setup_paths


%% compute reachability, load_graph takes as input the name of a temporal graph
[~,E] = load_graph('enron-only');
S = reach_graph(E, 'enron-only.mtx');


%% compute reachability graph, tscc bounds, and stats
data = 'test';
compute_reach(data);

% or more generally:
data = 'graphs';
compute_reach(data);


%% computes network stats/tscc bounds from existing reach data
reach_data(data)

%% afterwards, run pmc, using the reachability graph as input
