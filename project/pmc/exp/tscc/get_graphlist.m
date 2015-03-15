function graphlist = get_graphlist(data)
% loads the appropriate graphlist into mem

if (strcmp(data,'retweets')),
    dynamic_retweets
elseif (strcmp(data,'fb')),
    dynamic_fb
elseif (strcmp(data,'test')),
	test_graph
else
    dynamic_graphs
end