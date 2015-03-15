clear all;
close all
threads = [1 2 4 8 16]
setupfigs

sym = {'-','--','-.','-.'};
colors = {rgb('Crimson'), rgb('DodgerBlue'), rgb('LimeGreen'), rgb('MediumPurple'), rgb('DeepPink'),...
     rgb('DeepSkyBlue'), rgb('OrangeRed')}

R = [ 290.88 		156.09 		91.84 		62.15 		46.3;   % soc-orkut
      22.8 		16.96 		11.08 		7.77 		4.87;       % soc-flickr
      2.37 		1.27 		0.71 		0.45 		0.29;]      % socfb-Texas
    

for i=1:length(threads),
   S(:,i) = R(:,1) ./ R(:,i)
end

close all
h = figure('Visible', 'off');
lz = 2;
sz = 16;
for i=1:size(S)
    plot(threads,S(i,:),'-o','Color',colors{i},'MarkerFaceColor',colors{i}, 'MarkerSize',5 , 'LineWidth',lz); hold on;
end
solvers = {}
solvers{1} = 'soc-orkut (290)';
solvers{2} = 'soc-flickr (22)';
solvers{3} = 'socfb-Texas (2.3)';

leg = legend(solvers,'location','NW');
set(leg,'fontsize',sz-4);
set(leg,'box','off');

xlabel('Processors', 'FontSize', sz);
ylabel('Speedup', 'FontSize', sz);
box off;
ylim([0 max(max(S))+0.5])
xlim([0 max(threads)])
set(gca,'XTick',[0 1 4 8 12 16])


fname = 'social';
save_figure(h,['output/speedup-',fname],'medium-ls');