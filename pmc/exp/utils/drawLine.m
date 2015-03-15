function drawLine(d,td,lg)

colors  = ['m' 'b' 'r' 'g' 'c' 'k' 'y'];
lines   = [':' '-' '-.' '--'];
markers = ['x' '*' 's' 'd' 'v' '^' 'o'];

[ns,np] = size(d);

rndclr=[rand, rand, rand];

hold on;

for p = 1: np
  option = ['-' colors(p) markers(p)];
  h(p) = plot(d(:,p),option,'MarkerSize',8,'LineWidth',3);
end

ylabel('Runtime (log scale)','FontSize',20);
xlabel('Ratio of clique size to optimal (%)','FontSize',20);
ul = max(max(d));
ll = min(min(d));

[ns1,np1] = size(td);

v = axis;
v(1) = 0;
v(2) = ns1 + 1;
v(3) = ll - 1;
v(4) = ul + 1;

axis(v);
set(gca,'XTick',1:size(td))
set(gca, 'XTickLabel',td, 'FontSize',16);

%set(gca,'XTick',1:size(td));
%set(gca, 'XTickLabel',td, 'FontSize',12);
legend(lg, 1, 'EdgeColor','w','orientation', 'vertical','Location','Best');

hold off;

