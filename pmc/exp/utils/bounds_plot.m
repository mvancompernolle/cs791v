%% read data
fid=fopen('static-graphs-reclassify.csv'); C = textscan(fid, '%s%f%f%f%f%f%f%f%f%f%f%s%s%f','Delimiter',','); fclose(fid);

%% prepare bounds
n = C{2};
e = C{3};
maxd = C{4};
avgd = 2*e./n;
kbar = C{6};
maxt = C{7};
avgt = C{8};
sqrt2T = C{9};
K = C{10};
omega = C{11};
types = nominal(C{12});
dtypes = double(types);
types2 = nominal(C{13});
flag = logical(C{14});

%% colors
colors = [27, 158, 119; 217, 95, 2; 117, 112, 179; 231, 41, 138; 102, 166, 30; 230, 171, 2; 166, 118, 29; 102, 102, 102; ]/255;

%%
Cflag = C{1}(flag);
tflag = dtypes(flag);
maxdf = maxd(flag);
avgdf = avgd(flag);
[~,p] = sort(maxd(flag)./omega(flag));
%B = [(K+1)./omega (sqrt2T+1)./omega avgd./omega avgt./((omega-1).*(omega-2)/2)];
B = [(K+1)./omega (sqrt2T+1)./omega maxd./omega];
B = B(flag,:);
clf;
for i=[1 2 5 10*[1 2 5] 100*[1 2 5] 1000]
    line([-0.2,numel(p)], [log10(i), log10(i)], 'Color', 0.75*[1,1,1],'LineWidth', 0.4);
    %line([-0.2 0.2], [i/10 i/10],'Color','k')
    text(-0.4, log10(i)-0.007, num2str(i),'HorizontalAlignment','right','VerticalAlignment','middle');
end
ylim([-1.5 4])

hold on;
plot(log10(B(p,:)),'.','MarkerSize',8);
box off;
axis off;

for i=1:numel(p)
    text(i,min(log10(B(p(i),:)))-0.05,Cflag(p(i)) ,'Rotation',90,'HorizontalAlignment','right','FontSize',9, 'Color',colors(tflag(p(i)),:));
    text(i,max(log10(B(p(i),:)))+0.1,sprintf('%.1f',maxdf(p(i))/avgdf(p(i))),'FontSize', 9, 'Rotation', 90);
end

%set(gcf,'Color',0.9*[1,1,1]);
%set(gcf,'InvertHardcopy','off');

pp = get(gcf,'PaperPosition');
pp([3,4]) = [6.5,2.5];
set(gcf,'PaperPosition',pp);

print(gcf,'cbounds-final.eps','-depsc2');


%%
% -- old plot (a) will have max-clique vs global cc, instead of
% max_clique/max_kcore vs global cc

clf;

scatter(kbar,log10(omega),20,dtypes,'Filled');
colormap(colors);

showflag = [
65 %   yeast 
15 %   celegans 
15 %   enron-only 
70 %   wiki-talk 
60 %   enron-large 
0 %   fb-messages 
70 %   reality 
55 %   infect-hyper 
65 %   infect-dubli 
0 %   wiki-vote 
0 %   epinions 
34 %   youtube 
65 %   slashdot 
0 %   flickr 
0 %   orkut 
45 %   livejournal 
0 %   gowalla 
0 %   brightkite 
0 %   duke14 
0 %   berkeley13 
0 %   penn94 
15 %   stanford3 
75 %   texas84 
0 %   p2p-gnutella 
0 %   internet-as 
0 %   routers-rf 
45 %   whois 
45 %   as-skitter 
45 %   mathscinet 
45 %   ca-condmat 
45 %   ca-astroph 
45 %   ca-hepph 
15 %   polblogs 
65 %   web-google 
0 %   wikipedia
0 %   retweet 
10 %   twitter-cope 
90 %   retweet-craw 
];

for i=1:numel(n)
    if showflag(i)
        text(kbar(i),log10(omega(i))+0.05,C{1}(i), ...
            'Rotation',showflag(i),'FontSize',7,'Color',colors(dtypes(i),:));
    end
end

xlabel('clustering coefficient');
ylabel('log of clique size');
pp = get(gcf,'PaperPosition');
pp([3,4]) = [3,3];
set(gcf,'PaperPosition',pp);
print(gcf,'omega-vs-cc.eps','-depsc2');

%%
clf;

scatter(log10(avgt),log10(omega),20,double(dtypes),'Filled');
colormap(colors);

xlabel('log of average triangle count per node');
ylabel('log of clique size');

showflag = [
65 %   yeast 
1 %   celegans 
0 %   enron-only 
70 %   wiki-talk 
60 %   enron-large 
0 %   fb-messages 
70 %   reality 
45 %   infect-hyper 
0 %   infect-dubli 
90 %   wiki-vote 
0 %   epinions 
0 %   youtube 
0 %   slashdot 
0 %   flickr 
0 %   orkut 
45 %   livejournal 
0 %   gowalla 
0 %   brightkite 
0 %   duke14 
0 %   berkeley13 
0 %   penn94 
1 %   stanford3 
45 %   texas84 
90 %   p2p-gnutella 
0 %   internet-as 
0 %   routers-rf 
45 %   whois 
45 %   as-skitter 
45 %   mathscinet 
65 %   ca-condmat 
65 %   ca-astroph 
45 %   ca-hepph 
35 %   polblogs 
0 %   web-google 
90 %   wikipedia
0 %   retweet 
1 %   twitter-cope 
90 %   retweet-craw 
];

for i=1:numel(n)
    if showflag(i)
        text(log10(avgt(i)),log10(omega(i))+0.05,C{1}(i), ...
            'Rotation',showflag(i),'FontSize',7,'Color',colors(dtypes(i),:));
    end
end

pp = get(gcf,'PaperPosition');
pp([3,4]) = [3,3];
set(gcf,'PaperPosition',pp);

print(gcf,'omega-vs-avgt.eps','-depsc2');


%%
clf;

scatter(log10(avgd),log10(omega),20,double(dtypes),'Filled');
colormap(colors);

xlabel('log of average degree');
ylabel('log of clique size');

showflag = [
65 %   yeast 
1 %   celegans 
0 %   enron-only 
0 %   wiki-talk 
30 %   enron-large 
25 %   fb-messages 
80 %   reality 
45 %   infect-hyper 
30 %   infect-dubli 
0 %   wiki-vote 
0 %   epinions 
0 %   youtube 
0 %   slashdot 
25 %   flickr 
45 %   orkut 
45 %   livejournal 
0 %   gowalla 
0 %   brightkite 
0 %   duke14 
0 %   berkeley13 
0 %   penn94 
45 %   stanford3 
0 %   texas84 
0 %   p2p-gnutella 
0 %   internet-as 
0 %   routers-rf 
45 %   whois 
45 %   as-skitter 
58 %   mathscinet 
80 %   ca-condmat 
65 %   ca-astroph 
45 %   ca-hepph 
35 %   polblogs 
0 %   web-google 
90 %   wikipedia
0 %   retweet 
1 %   twitter-cope 
0 %   retweet-craw 
];

for i=1:numel(n)
    if showflag(i)
        text(log10(avgd(i)),log10(omega(i))+0.05,C{1}(i), ...
            'Rotation',showflag(i),'FontSize',7,'Color',colors(dtypes(i),:));
    end
end
xlim([0,2.1]);
pp = get(gcf,'PaperPosition');
pp([3,4]) = [3,3];
set(gcf,'PaperPosition',pp);

print(gcf,'omega-vs-avgd.eps','-depsc2');