clear all;
close all;
threads = [1 2 4 8 16]
datapath = 'mcdata/';
prefix = 'speedup_'
name = 'mc_results_sample_graphs_';
suffix = 'threads.txt';

R = dlmread([datapath,prefix,name,suffix]);
R(:,6) = []

who
setupfigs

sym = {'-','--','-.','-.'};
colors = {rgb('Crimson'), rgb('DodgerBlue'), rgb('LimeGreen'), rgb('MediumPurple'), rgb('DeepPink'),...
     rgb('DeepSkyBlue'), rgb('OrangeRed')}


for i=1:length(threads),
   S(:,i) = R(:,1) ./ R(:,i)
end

SL = [S(2,:); S(4,:); S(11:15,:);];
S = SL;


close all
h = figure('Visible', 'off');
lz = 2;
sz = 16;
for i=1:size(S)
    plot(threads,S(i,:),'-o','Color',colors{i},'MarkerFaceColor',colors{i}, 'MarkerSize',5 , 'LineWidth',lz); hold on;
end
    plot(threads,threads,'--','Color',rgb('Black'), 'LineWidth',1); hold on;
hold off;

fname = 'dimacs';

solvers{1} = 'brock400-4 (331)';
solvers{2} = 'san200-0-9-2 (1)';
solvers{3} = 'san400-0-7-1 (0.2)';
solvers{4} = 'brock800-4 (3604)';
solvers{5} = 'brock400-3 (619)';
solvers{6} = 'p-hat1500-1 (4)';
solvers{7} = 'san1000 (1)';

leg = legend(solvers,'location','NW');
set(leg,'fontsize',sz-4);
set(leg,'box','off');

xlabel('Processors', 'FontSize', sz);
ylabel('Speedup', 'FontSize', sz);

ylim([0 20])
set(gca,'XTick',[0 1 4 8 12 16])
xlim([0 16])

save_figure(h,['output/speedup-',fname],'medium-ls');