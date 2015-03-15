function compute_graph_data(data, prefix)
% compute graph statistics and bounds for max clique
%
% David Gleich, Ryan Rossi
% Copyright 2012 
%
%

setup_paths

if nargin < 2,
   prefix = ''; 
end

if nargin == 0,
    data = 'graphs';
end

% load the list of graphs
graphlist = get_graphlist(data);

% compute stats and save to container for later
results_path = '../results/';
mkdir([results_path,data])
gdata = containers.Map();
for i=1:size(graphlist,1)
    ginfo = struct();
    A = load_graph(graphlist{i,2});
    
    ginfo.name = graphlist{i,2};
    ginfo.name
    assert(all(diag(A)==0));
    ginfo.nverts = size(A,1);
    ginfo.nedges = nnz(A)/2;
    ginfo.degrees = full(sum(A,2));
    ginfo.maxdeg = max(ginfo.degrees);
    ginfo.avgdeg = mean(ginfo.degrees);
    
    
    tic; [~,~,~,cc,t]=triangleclusters(A); toc;
    ginfo.triangles = t;
    ginfo.clustercoef = cc;
    ginfo.avgcc = mean(cc);
    ginfo.tavg = mean(t);
    ginfo.maxtriangles = max(t);
    
    
    %triangle upperbound on mc size: tri_bound+1
    ginfo.tri_bound = ceil(sqrt(2*max(t))); %todo: label as ub
   
    % kcore upperbound on mc size: max_kcore+1
    tic; ginfo.cores = core_numbers(A); toc;
    ginfo.maxcore = max(ginfo.cores); 
    
    
    ginfo.type = graphlist{i,4};
    ginfo.edgetype = graphlist{i,3};
    
    gdata(ginfo.name) = ginfo;
    ginfo
    
    save(['../results/',data,'/gdata'], 'gdata', '-v7.3');
end

export2mtx(data, prefix)
