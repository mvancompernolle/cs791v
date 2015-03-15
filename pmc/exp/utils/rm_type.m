function str = rm_type(str)
% remove type tags

types = {'bio-','ca-','cit-','soc-','ia-','tech-','rec-'}; %,'web-'};

for i=1:length(types),
   str = strrep(str,types{i},''); 
end