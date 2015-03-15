function sgdata = cont2struct(gdata)
% convert container into struct

for graph=gdata.keys
    if ~exist('sgdata','var'), sgdata = gdata(graph{1});
    else, sgdata(end+1) = gdata(graph{1}); end
end