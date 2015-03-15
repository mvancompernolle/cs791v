
clear all
close all
datapath = 'mcdata/pmc_baseline_pp_plot.txt';

% each column is a solvers performance
S = dlmread(datapath)

setupfigs

solvers{1} = 'pmc (no neigh cores)';
solvers{2} = 'pmc';
solvers{3} = 'BK';
solvers{4} = 'FMC';


folder = 'output/'
type = 'serial_baselines'
fn = [folder,type,'_log'];

perf_profile(S,solvers,fn,1)  % log plot



