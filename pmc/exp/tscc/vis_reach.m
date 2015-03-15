function vis_reach(data)

setup_paths
if nargin == 0, data = 'graphs'; end

graphlist = get_graphlist(data);

exportpath = '../results/vis_reach/';
mkdir([exportpath])
for i=1:size(graphlist,1)
    A = load_reach([graphlist{i,2},'_scc']);
    fprintf('%s\n',graphlist{i,2});
    export2graphml(A,[exportpath,graphlist{i,2},'_scc','.graphml']);
end

