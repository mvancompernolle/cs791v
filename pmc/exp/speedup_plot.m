function [] = speedup_plot(P, solvers, fn)

setupfigs
sym = {'-','--','-.','-.'};
colors = {rgb('Crimson'), rgb('DodgerBlue'), rgb('LimeGreen'), rgb('MediumPurple'), rgb('DeepPink'),...
     rgb('DeepSkyBlue'), rgb('OrangeRed')}


for i=1:length(threads),
   S(:,i) = P(:,1) ./ P(:,i)
end

S

close all
h = figure('Visible', 'off');
lz = 2;
sz = 16;
for i=1:size(S)
    plot(threads,P(i,:),'-o','Color',colors{i},'MarkerFaceColor',colors{i}, 'MarkerSize',5 , 'LineWidth',lz); hold on;
end
plot(threads,threads,'--','Color',rgb('Black'), 'LineWidth',1); hold on;
hold off;


leg = legend(solvers,'location','NW');
set(leg,'fontsize',sz-4);
set(leg,'box','off');

xlabel('Processors', 'FontSize', sz);
ylabel('Speedup', 'FontSize', sz);


ylim([0 max(max(S))+0.5])
xlim([0 max(threads)])


save_figure(h,['plots/speedup-',fn],'medium-ls');
