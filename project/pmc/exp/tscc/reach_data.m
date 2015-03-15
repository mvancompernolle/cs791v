function rdata = reach_data(data)
% computes network stats and tscc upperbounds
% 
% reach graphs must already be computed,
% if not, see compute_reachability.m
%
% David Gleich, Ryan Rossi, 
% Copyright 2012, Purdue University
%

setup_paths
if nargin == 0, data = 'graphs'; end
graphlist = get_graphlist(data);

results_path = '../results/';
mkdir([results_path,data]);
rdata = containers.Map();
for i=1:size(graphlist,1)
    rinfo = struct();
    [A E] = load_graph(graphlist{i,2});
    rinfo.name = graphlist{i,2};
    
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
    
    S = load_reach([graphlist{i,2},'_scc']);
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