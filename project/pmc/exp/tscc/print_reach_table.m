function print_reach_table(data)
% prints largest tscc (mc), upperbounds, and other stats
%
% Ryan Rossi, Purdue University
% Copyright 2012
%

setup_paths

if nargin == 0, data = 'graphs'; end
graphlist = get_graphlist(data);

update_tscc(data)

dpath = ['../results/',data];
load([dpath,'/rdata'])
sgdata = cont2struct(rdata);


header = {
'\begin{table*}[h!]';
'\caption{Network statistics and bounds for the dynamic graphs and their strong reachability graphs.}';
'\label{table:reach-data}';
'\centering\small';
'\begin{tabularx}{\linewidth}{ ll XXXXXX XXc c}';
'\toprule';
'\textbf{Graph} & & $|V|$ & $|E|$ & ';
'$d_{\max}$ & $d_{\text{avg}}$ & $\mathbf{\bar{\kappa}}$ & '
'$T$ & $T_{\text{avg}}$ & $\sqrt{2T}$ & $K$ & $\omega$\\'
'\midrule';
};
for i=1:length(header), fprintf('%s\n',header{i}); end

max_strsize = 12;
print_ginfo = @(g) ...
       fprintf('\\multirow{2}{*}{\\rotatebox{0}{\\textsc{%s}}}\n \t & $\\rm G$ & %d & %d & %d & %0.1f & %0.2f & %.0f & %.1f & %.0f & %d & \\textbf{%d}\\\\ \n', ...
            lower(strrep(strrep(strrep(substr(g.name,1,max_strsize), '_', '-'),'2009',''),'rt-', '\scriptsize \#')),...
                g.nverts, g.nedges,...
                max(g.degrees), mean(g.degrees), mean(g.clustercoef), ...
                max(g.triangles), mean(g.triangles), sqrt(2*max(g.triangles)), g.maxcore, g.maxclique); 


print_ginfoRs = @(g) ...
            fprintf('\t & $\\mR_s$ & %d & %d & %d & %0.1f & %0.2f & %.0f & %.1f & %.0f & %d & \\textbf{%d}\\\\ \n', ...
                g.rs.nverts, g.rs.nedges,...
                max(g.rs.degrees), mean(g.rs.degrees), mean(g.rs.clustercoef), ...
                max(g.rs.triangles), mean(g.rs.triangles), sqrt(2*max(g.rs.triangles)), g.rs.maxcore, g.rs.maxclique); 
            
min_rs_edges = 60;
types = unique({sgdata.type});
for ti=1:numel(types)
    type = types{ti};
    tgraphs = sgdata(strcmp({sgdata.type},type)); 
    [~,p] = sort([tgraphs.nverts]); 
    for gi=1:numel(p)
        if tgraphs(p(gi)).rs.nverts >= min_rs_edges,
            print_ginfo(tgraphs(p(gi)));
            print_ginfoRs(tgraphs(p(gi)));
            fprintf('\\midrule \n');
        end
    end
end
fprintf('\\end{tabularx} \n\\end{table*} \n');
