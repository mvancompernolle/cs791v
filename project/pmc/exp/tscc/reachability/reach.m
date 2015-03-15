function R = reach(E, is_und)
% Compute reachability graph (or temporal path matrix)
%
% input:    
%           E:    temporal edge list, (u, v, t)
%
% output:
%           R:    1 if u can temporally reach v
%                 (if there exists a temporal path from u to v)
%
%
% Ryan Rossi, David Gleich,
% Purdue University, Copyright 2012
%

if nargin < 1, error('edge list undefined.'); end;
if nargin < 2, 
    is_und = 0; 
end

cols = size(E,2);
if cols < 3,
    fprintf('E must have at least 3 columns');
end
if cols > 3,
    fprintf('E has %d columns, expected 3 columns [u v t] \n',cols);
    fprintf('ignoring additional columns \n');
end

node_ids = unique(E(:,1:2)); 
n = max(node_ids);
m = size(E,1);
fprintf('dynamic graph:  %d vertices, %d edges \n', n, m);

[~,idx] = sort(E(:,3),'descend'); % ensure reverse time order
E = E(idx,:);

R = speye(n);
E=E';

tic
if ~is_und,
    for e_idx=1:m,
        R(:,E(1,e_idx)) = R(:,E(1,e_idx))|R(:,E(2,e_idx));
    end
else
    for e_idx=1:m,
        R(:,E(1,e_idx)) = R(:,E(1,e_idx))|R(:,E(2,e_idx));
        R(:,E(2,e_idx)) = R(:,E(1,e_idx)); 
    end
end
toc

R = R - diag(diag(R));
R = R'; 

fprintf('reach.m >> reachability graph: %d vertices, %d edges \n',size(R,1), nnz(R));
