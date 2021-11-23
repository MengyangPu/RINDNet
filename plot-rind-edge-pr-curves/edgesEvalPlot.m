function edgesEvalPlot(path, names, colors, lines, years, human)
% Plot edge precision/recall results for directory of edge images.
%
% USAGE
%  edgesEvalPlot( path, names, [colors], [lines], [years] )
%
% INPUTS
%  path        - algorithm result directory
%  names       - {nx1} algorithm names (for legend)
%  colors      - [{nx1}] algorithm colors
%  lines       - [{nx1}] line styles
%  years       - [{nx1}] the years when the algorithms are proposed
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parse inputs
if(~iscell(names)), names={names}; end
if(nargin<3||isempty(colors)), colors=repmat({'r','g','b','k','m','c','y'},1,100); end
%if(nargin<4||isempty(lines)), lines=repmat({'-'},1,100); end
if(nargin<5||isempty(years)), years=repmat({''},1,100); end
if(~iscell(colors)), colors={colors}; end
if(~iscell(lines)), lines={lines}; end
if(~iscell(years)), years={years}; end

% setup basic plot (isometric contour lines and human performance)
clf; box on; grid on; hold on;
%line([0 1],[.5 .5],'LineWidth',2,'Color',.7*[1 1 1]);
for f=0.1:0.1:0.9, r=f:0.01:1; p=f.*r./(2.*r-f); %f=2./(1./p+1./r)
  plot(r,p,'Color',[0 1 0]); plot(p,r,'Color',[0 1 0]); end
%if(human), h=plot(0.7235,0.9014,'o','MarkerSize',8,'Color',[0 .5 0],...
%  'MarkerFaceColor',[0 .5 0],'MarkerEdgeColor',[0 .5 0]); end
set(gca,'XTick',0:0.1:1,'YTick',0:0.1:1);
grid on; xlabel('Recall'); ylabel('Precision');
axis equal; axis([0 1 0 1]);

% load results for every algorithm (pr=[T,R,P,F])
n=length(names); hs=zeros(1,n); res=zeros(n,9); prs=cell(1,n);
for i=1:n
  pr=dlmread(fullfile(path,[names{i} '_bdry_thr.txt'])); pr=pr(pr(:,2)>=1e-3,:);
  [~,o]=unique(pr(:,3)); R50=interp1(pr(o,2),pr(o,1),max(pr(o(1),3),.5));
  res(i,1:8)=dlmread(fullfile(path,[names{i} '_bdry.txt'])); res(i,9)=R50; prs{i}=pr;
end

% sort algorithms by ODS score
[~,o]=sort(res(:,4),'descend'); res=res(o,:); prs=prs(o);
colors=colors(o);  lines=lines(o);  names=names(o); years=years(o);

% plot results for every algorithm (plot best last)
for i=n:-1:1
  hs(i)=plot(prs{i}(:,2),prs{i}(:,3),'-','LineWidth',2,'Color',colors{i},'LineStyle',lines{i});
  fprintf('ODS=%.3f OIS=%.3f AP=%.3f R50=%.3f',res(i,[4 7:9]));
  if(~isempty(names)), fprintf(' - %s',[names{i} years{i}]); end; fprintf('\n');
end

% show legend if nms provided (report best first)
hold off; if(isempty(names)), return; end
for i=1:n, names{i}=sprintf('[F=.%3d] %s',round(res(i,4)*1000),[names{i} years{i}]); end
legend(hs,names,'FontSize',13,'Location','sw');

end
