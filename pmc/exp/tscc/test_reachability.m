

%% Test Case 1
% Temporal SCC = {3-4-1} or {3-4-2} 
% max cliques in reachability graph R
%

% (i, j, time)
E = [
    3 4 1;
    4 1 2;
    1 3 3;
    3 4 4;
    4 2 5;
    2 3 6;
    3 4 7;
];


R_actual =  ...
[0     1     1     1;
 0     0     1     1;
 1     1     0     1;
 1     1     1     0];


R = reach(E);


if all(all(R==R_actual)) == 0, fprintf('Error: R graph is incorrect!')
else fprintf('Test passed! (R_actual == R_out) \n');
end

full(R) 
R_actual



%% Test Case 2
%

E = [
    1 2 1;
    2 3 7;
    3 4 5;
    4 5 3;
    5 4 2;
    4 3 8;
    3 2 6;
    2 1 4;
];

R = reach(E);

R_actual = [
  0 1 1 0 0;
  1 0 1 0 0;
  0 1 0 1 0;
  0 0 1 0 1;
  0 0 1 1 0;
];

if all(all(R==R_actual)) == 0, fprintf('Error: R graph is incorrect!')
else fprintf('Test passed! (R_actual == R_out) \n');
end

full(R) 
R_actual



%% Test Case 3 (complete test)
%

E = [
    2 3 1;
    1 2 2;
    3 1 3;
    7 6 4;
    6 4 5;
    4 5 6;
    5 7 7;
    7 6 8;
    1 4 9;
];

R = reach(E);

R_actual = [
  0 1 0 1 0 0 0;
  1 0 1 1 0 0 0;
  1 0 0 1 0 0 0;
  0 0 0 0 1 1 1;
  0 0 0 0 0 1 1;
  0 0 0 1 1 0 1;
  0 0 0 1 1 1 0;
];

if all(all(R==R_actual)) == 0, fprintf('Error: R graph is incorrect!')
else fprintf('Test passed! (R_actual == R) \n');
end

R = full(R) 
R_actual




%% Test Strong Reachability: 
%

Rs = strong_reach(R)
Rs_actual = R_actual&R_actual'

if all(all(Rs==Rs_actual)) == 0,
     fprintf('Error: strong reachability graph Rs is incorrect!')
else fprintf('Test passed! (Rs_actual == Rs) \n'); 
end


%% Write edges
%
write_mtx(Rs,'testing_R.mtx');

%expected edges
[src dest val] = find(tril(Rs));
edges = [src dest val]

