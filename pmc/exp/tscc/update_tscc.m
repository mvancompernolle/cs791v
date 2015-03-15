function gdata = update_tscc(data)
% update tscc of Rs in container,
%   also add maxclique of G
% Both are used later w/ bounds
%
% Ryan A. Rossi, Purdue University
% Copyright 2012

setup_paths
results_path = '../results/';

if nargin == 0, data = 'graphs'; end
graphlist = get_graphlist(data);

% get maxclique of G
actual_mc = ['../../mc/results/','fmc/',data,'/'];
info_path = [actual_mc,'info_',data,'_mc.txt'];
files = find_files(info_path);
if isempty(files),
    info_path = [actual_mc,'info_graphs_scc.txt'];
    files = find_files(info_path);
end
fid = fopen(info_path);

line = fgetl(fid);
gdata = containers.Map();
while ischar(line),
    d = strsplit(line,',');
    if isKey(gdata,d{3}),
        ginfo = gdata(d{3});
        if strcmp(d{2},'exact'),
            ginfo.maxclique = str2double(d{4});
        elseif strcmp(d{2},'heur'),
            ginfo.heur_maxclique = str2double(d{4});
        end
        gdata(d{3}) = ginfo;
    else
        ginfo = struct();
        ginfo.name = d{3};
        if strcmp(d{2},'exact'),
            ginfo.maxclique = str2double(d{4});
        elseif strcmp(d{2},'heur'),
            ginfo.heur_maxclique = str2double(d{4});
        end
        gdata(d{3}) = ginfo;
    end
    line = fgetl(fid);
end
fclose(fid);


load([results_path,data,'/rdata']);
for graph=rdata.keys,
    ginfo = rdata(graph{1});
    if ~isfield('maxclique',ginfo) && ~isfield('heur_maxclique',ginfo),
        ginfo.maxclique = 0; ginfo.heur_maxclique = 0;
    end
    rinfo = ginfo.rs;
    if ~isfield('maxclique',rinfo) && ~isfield('heur_maxclique',rinfo),
        rinfo.maxclique = 0; rinfo.heur_maxclique = 0;
    end
    ginfo.rs = rinfo;
    rdata(graph{1}) = ginfo;
end


% largest temporal scc of R_s
mcpath = [results_path,'fmc/',data,'/'];
info_path = [mcpath,'info_',data,'_tscc.txt'];
files = find_files(info_path);
if isempty(files),
    info_path = [mcpath,'info_graphs_mc.txt'];
    files = find_files(info_path);
end
fid = fopen(info_path);



line = fgetl(fid);
while ischar(line)
    d = strsplit(line,',');
    key = d{3};
    if strcmp(d{2},'exact'),
        if isKey(rdata,key),
            ginfo = rdata(key); 
            if isKey(gdata,key),
                ginfo.maxclique = gdata(key).maxclique;
                ginfo.heur_maxclique = gdata(key).heur_maxclique;
            end
            rinfo = rdata(key).rs;
            rinfo.maxclique = str2double(d{4});
            ginfo.rs = rinfo;
            rdata(key) = ginfo;
        end
    end
    if strcmp(d{2},'heur'),
        if isKey(rdata,key),
            ginfo = rdata(key); 
            if isKey(gdata,key),
                ginfo.heur_maxclique = gdata(key).heur_maxclique;
                if ginfo.maxclique == 0, ginfo.maxclique = ginfo.heur_maxclique; end
            end
            rinfo = rdata(key).rs; 
            rinfo.heur_maxclique = str2double(d{4});
            ginfo.rs = rinfo;
            rdata(key) = ginfo;
        end
    end
    line = fgetl(fid);
end
fclose(fid);


for graph=rdata.keys
    ginfo = rdata(graph{1});
    if ginfo.maxclique == 0, ginfo.maxclique = ginfo.heur_maxclique; end
    rinfo = ginfo.rs;
    if rinfo.maxclique == 0, rinfo.maxclique = rinfo.heur_maxclique; end
    ginfo.rs = rinfo;
    rdata(graph{1}) = ginfo;
end

save([results_path,data,'/rdata.mat'], 'rdata')