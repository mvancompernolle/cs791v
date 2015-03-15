function types = graph_types(type_names)
% todo: update for all graph types

info = {
    'soc','social networks';
    'soc-fb','facebook networks';
    'bio','biological networks';
    'ca','collaboration networks';
    'interaction','interaction networks';
    'rt','retweet networks';
    'tech','technological networks';
    'web','web graphs';
    'rec','recommendation networks';
    'cit','citiation networks';
    };

for i=1:length(type_names),
    name = type_names{i};
    for j=1:length(info),
        if strcmp(name,info{j,1}),
            types{i} = info{j,2};
            break;
        end
    end
end
