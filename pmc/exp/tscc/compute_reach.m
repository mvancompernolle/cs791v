function rdata = compute_reach(data)
% computes reachability graph, upperbounds for the size of tscc, 
%   and other stats
% 
% Example:
%   compute_reach('graphs')
%
% The above computes reachability for each graph in the graphlist.
%
% 'graphs'    : dynamic_graphs.m
% 'retweets'  : dynamic_retweets.m
%

% David Gleich, Ryan Rossi, 
% Copyright 2012, Purdue University

setup_paths
if nargin == 0, data = 'graphs'; end;
graphlist = get_graphlist(data);


results_path = '../results/';
mkdir([results_path,data]);
rdata = containers.Map();
for i=1:size(graphlist,1)
    rinfo = struct();
    [A E] = load_graph(graphlist{i,2});
    rinfo.name = graphlist{i,2};
    rinfo.src = 'local';

    
    rinfo.name
    assert(all(diag(A)==0));
    rinfo.nverts = size(A,1);
    rinfo.nedges = size(E,1);
    rinfo.degrees = full(sum(A,2));
    rinfo.maxdeg = max(rinfo.degrees);
    rinfo.avgdeg = mean(rinfo.degrees);
    
    tic; [~,~,~,cc,t]=triangleclusters(A); toc;
    rinfo.triangles = t;
    rinfo.clustercoef = cc;
    rinfo.avgcc = mean(cc);
    rinfo.maxtriangles = max(t);
    rinfo.tavg = mean(t);
    rinfo.tri_bound = sqrt(2*max(t));
    rinfo.cores = core_numbers(A);
    rinfo.maxcore = max(rinfo.cores);
    rinfo.gr = 'G'; rinfo
    
    if strcmp(graphlist{i,3},'undirected'), is_und = 1;
    else is_und = 0; end
    
    S = reach_graph(E, [rinfo.name,'.mtx'], is_und);
    
    rs = struct();
    S = rm_zero_deg_verts(S);
    rs.nverts = size(S,1);
    rs.nedges = nnz(S)/2;
    rs.degrees = full(sum(S,2));
    rs.maxdeg = max(rs.degrees);
    rs.avgdeg = mean(rs.degrees);
    
    tic; [~,~,~,cc,t]=triangleclusters(S); toc;
    rs.triangles = t;
    rs.clustercoef = cc;
    rs.avgcc = mean(cc);
    rs.maxtriangles = max(t);
    rs.tavg = mean(t);
    rs.tri_bound = sqrt(2*max(t)); 
    rs.cores = core_numbers(S);
    rs.maxcore = max(rs.cores);   
    rs.gr = 'Rs'
    rinfo.rs = rs;
    
    rinfo.type = graphlist{i,4};
    rinfo.edgetype = graphlist{i,3};
    rdata(rinfo.name) = rinfo;
    rinfo
    
    save(['../results/',data,'/rdata'], 'rdata');
end