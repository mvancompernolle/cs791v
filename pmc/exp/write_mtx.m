function write_mtx(A, fname, out_path)
% Write edges to a file in the symmetric Matrix Market file format
% Edges should only be specified in the lower triangular
%
% INPUT:
%           A :   n x n symmetric matrix,
%                           *OR*
%                 m x 3 list of edges (must be lower triangular)
%
%
%   Input Format
%   ---------------
%   Links: http://math.nist.gov/MatrixMarket/formats.html#MMformat
%
%
%   Sample file:
%   ------------
%   Some specific commands related to mtx file format
%   #of r #of comlumns #of nonzeros
%   each line specifies row-index and column-index for each nonzero
%
%   For example:
%   ------------
%   %%MatrixMarket matrix coordinate pattern symmetric
%   8 8 6
%   3 2
%   4 1
%   5 2
%   5 3
%   5 4
%   8 4
%
%
% Ryan A. Rossi, Purdue University
% Copyright 2012-2013
%

if nargin < 3,
    out_path = '~/research/pmc/data/'; %todo: optional?
end

[r c] = size(A);
if r == c,
    A = tril(A);
    [src dest value] = find(A);
    E = [src dest value];
else
    E = A;
end


m = size(E,1); %num edges
unique_nodes = unique(E(:,1:2));
n = length(unique_nodes); %num unique nodes
max_nid = max(unique_nodes);


% write to .mtx file
fid = fopen(strcat(out_path,fname),'w+');
fprintf(fid,'%%MatrixMarket matrix coordinate pattern symmetric \n');
fprintf(fid,'%d %d %d\n',max_nid, max_nid, m);
fprintf(fid,'%d %d\n',E(:,1:2)');

fclose(fid);

