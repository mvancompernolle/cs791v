function export2graphml(A, fn)
% fn is a string,
% A is a binary symmetric adjmatrix
%
% Ryan A. Rossi, Purdue University
% Copyright 2012
%

fid = fopen (fn,'w');

fprintf (fid,'<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"\nxmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\nxsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns\nhttp://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n');
fprintf (fid,'<key id=\"d0\" for=\"node\" attr.type=\"double\">\n<default>0</default>\n</key>');
fprintf (fid,'<graph id=\"G\" edgedefault=\"undirected\">\n');

n = size(A,1);
for i=1:n
    fprintf (fid,'\t<node id=\"n%d\">\n<data key=\"d0\"></data>\n</node>\n',i-1);
end

[src dest val] = find(A);
E = [src dest];

fprintf(fid,'\t<edge source=\"n%d\" target=\"n%d\"/>\n',E(:,1:2)');

fprintf (fid,'</graph>\n');
fprintf (fid,'</graphml>');
fclose (fid);