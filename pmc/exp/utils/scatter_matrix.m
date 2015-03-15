function corr_info = scatter_matrix(X,names,fn,types,colors,type_names)
% Scatter matrix plots all correlations between columns of the X matrix
% Uses Pearson correlation by default, but can also use Spearman and Kendall
%
% Correlations can be highlighted in different ways
%
% Input
% ---------------------------------
%  X            n x m data matrix
%  names        m x 1 cell of labels for x and y axes
%  fn           string for eps filename
%  types        n x 1 vector of indices indicating type of row
%  colors       color assigned to each type, unique(colors) x 1 vector
%  type_names   str name for each type, same size as colors
%
% Output
% ----------------------------------
%  corr_info    info indicating sig correlations between vars
%
%
% Ryan A. Rossi
% Purdue University, Copyright 2012
%
method = 'Pearson'; alpha= 0.05;
corr_info = []; single_type = 0;

if nargin < 1, error('data matrix undefined.'); end
[n,m]= size(X);

if nargin < 2, names = 1:m; end
if nargin < 3, fn = 'corr-sc'; end
if nargin < 4, single_type = 1; end;

if m <= 4, psize=8; ssize=20; end
if m > 4, psize=14, ssize=24; end
if m > 10, psize=16, ssize=28; end

%fig = figure();
for j=1:m
    for k=1:j
        subaxis(m,m,(j-1)*m+k,'Holdaxis',0, ...
            'SpacingVertical',0.0005,'SpacingHorizontal',0,...
            'PaddingLeft',0,'PaddingRight',0,'PaddingTop',0,'PaddingBottom',0, ...
            'MarginLeft',.05,'MarginRight',.004,'MarginTop',.004,'MarginBottom',.07, ...
            'rows',[],'cols',[]);
        
        if j ~= k,
            [rho,pval] = corr(X(:,k),X(:,j),'type',method);
            if pval < alpha,
                sym = 'o'; color = [1 0 0];
                corr_info = [corr_info; k,j,rho,pval];
            else, sym = 'o'; color = [.5 .5 .5]; end
            
            if single_type == 1,
                plot(X(:,k),X(:,j),'.','MarkerSize',psize,'MarkerEdgeColor',color);
            else
                h = scatter(X(:,k),X(:,j),ssize,double(types),sym,'filled');
                colormap(colors)
                
                if pval < alpha, set(h,'MarkerEdgeColor','k'); end;
            end
        else
            x = X(:,j);
            x(isinf(x)) = 0;
            hist(x,5);
        end
        
        h = findobj(gca,'Type','patch');
        set(h,'FaceColor',[.2 .4 1],'EdgeColor','black')
        set (gca,'FontSize',6);
        %grid on
        
        
        if k == 1, ylabel(strcat('$$\mathbf{',names(j),'}$$'),'interpreter','latex','FontSize',16); end
        if j == m, xlabel(strcat('$$\mathbf{',names(k),'}$$'),'interpreter','latex','FontSize',16); end
        
        set(gca, 'XTickLabel',[], 'YTickLabel',[]); %todo: optional?
        
    end
end



%colors = colors(end:-1:1,:); %reverse order
if nargin > 5, %optional?
    dy_rec = 0.04;   dy_text = 0.04;
    y_rec  = 0.962;    y_text = 0.972; y_prev = y_text;
    for i=1:length(type_names),
        
        y_rec = y_rec - dy_rec;                 %[x y w h]
        annotation(gcf,'rectangle',...          % Create rectangle
            [0.69 y_rec 0.04 0.028],...
            'FaceColor',colors(i,:));
        
        
        y_text = y_text - dy_text;              %[x y w h]
        annotation(gcf,'textbox',...            % Create textbox
            [0.735 y_text 0.13 0.02],...
            'String',type_names{i},...
            'FontSize',14,...
            'FontWeight','bold',...
            'FontName','Monaco',...
            'FitBoxToText','on',...
            'EdgeColor','none');
    end
end

% annotation(gcf,'textbox',...            % Create textbox
%     [0.2 y_prev 0.13 0.02],...
%     'String','significant correlations are bolded',...
%     'FontSize',12,...
%     'FontWeight','bold',...
%     'FontName','Consolas',...
%     'FitBoxToText','on',...
%     'EdgeColor','none');



pp = get(gcf,'PaperPosition');
if m > 8,
    pp([3,4]) = [12,8];
    set(gcf,'PaperPosition',pp);
else
    %pp([3,4]) = [8,6];
    set(gcf,'PaperPositionMode','auto')
end
    
print(gcf,fn,'-depsc2','-r300','-painters')
%print(gcf,fn,'-depsc2','-painters');

