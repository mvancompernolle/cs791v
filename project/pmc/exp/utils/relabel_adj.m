function A = relabel_adj(A)


in_sum = find(sum(A) == 0);
out_sum = find(sum(A') == 0);
remove = intersect(in_sum,out_sum);
A(remove,:) = [];
A(:,remove) = [];