clear all;
close all;
threads = [1 2 4 8 16]
datapath = 'mcdata/';
prefix = 'table_perf_profile_';
suffix = 'threads.txt';

% each column is a solvers performance
T = {}
for i=1:length(threads),
    fprintf([datapath,prefix,num2str(threads(i)),suffix,'\n'])
    T{i} = dlmread([datapath,prefix,num2str(threads(i)),suffix]);
end

who

folder = 'output/'
name = 'dimacs'

perf_profile_labels


for i=1:length(threads),
    fn = [folder,'pmc_pp_',name,'_th',num2str(threads(i)),'_log'];
    size(T{i})
    perf_profile(T{i},solvers,fn,1)
end


