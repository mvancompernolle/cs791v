function [ ginfo ] = graph_info( fn )


load_libs

ginfo = struct();
A = load_graph(fn);
ginfo.name = fn;
ginfo.src = 'local';
ginfo.name
assert(all(diag(A)==0));
ginfo.nverts = size(A,1);
ginfo.nedges = nnz(A)/2;
ginfo.degrees = full(sum(A,2));
ginfo.maxdeg = max(ginfo.degrees);
ginfo.avgdeg = mean(ginfo.degrees);


tic
[cond cut vol cc t]=triangleclusters(A);
ginfo.triangles = t;
ginfo.cond = cond;
ginfo.cut = cut;
ginfo.vol = vol;
ginfo.clustercoef = cc;
ginfo.avgcc = mean(cc);
ginfo.maxtriangles = max(t);
ginfo.tri_bound = sqrt(max(t)); %upper-bound on maxclique size
toc

tic
ginfo.cores = core_numbers(A);
ginfo.maxcore = max(ginfo.cores);
toc


tic
ginfo.ncomps = max(scomponents(A));
ginfo.maxcomp = size(largest_component(A),1);
toc


gdata(ginfo.name) = ginfo;
ginfo



