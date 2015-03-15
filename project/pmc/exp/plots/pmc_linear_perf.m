clear all
threads = [1 2 4 8 16]
name = 'mcdata/pmc_linear_runtime.txt';
R = dlmread(name);
who

sym = {'-','--','-.','-.'};
colors = {rgb('Crimson'), rgb('DodgerBlue'), rgb('MediumPurple'), rgb('DeepPink'), rgb('LimeGreen'), rgb('DeepSkyBlue'), rgb('OrangeRed')}

close all
h = figure('Visible', 'off');
lz = 2;
sz = 16;

R = log10(R)

plot(R(:,1),R(:,2),'o','Color',colors{2},'MarkerFaceColor',...
        colors{2}, 'MarkerSize',3 , 'LineWidth',lz); hold on;

ln_h = lsline

set(ln_h(1),'color','black')
set(ln_h(1),'LineWidth',1)

fname = 'output/pmc_linear_runtime';

xlabel('|V| + |E|', 'FontSize', sz);
ylabel('Runtime', 'FontSize', sz);

save_figure(h,['',fname],'medium-ls');

