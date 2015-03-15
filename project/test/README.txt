BBMC (MaxCliqueSeq implementation) general information

The following assumes you are in a console in a directore where you heve copied/uncompressed the source files.

1. Compiling
    to compile sources run:
    make
    to clean (remove the binary) run:
    make clean
    
2. Running
    run the binary executable:
    ./bbmc <graph_file_name>
    where 
        graph_file_name : either relative or absolute path to the file containing an undirected graph in DIMACS file format (http://prolland.free.fr/works/research/dsat/dimacs.html)
    The results are reported in text form on the standard output, listed in the following form:
        Loading DIMACS graph <graph_file_name>
        <|V|> vertices <|E|> edges <graph_density>
	-- MC(int,bitstring based set,Graph,MCR sort for Bitstrings<greedy color sort with resort on bitstrings>)
        search took <t>; <s> steps
        Clique (<|Q|>): [Q1,Q2,...,Qq]
    where the variables filled by the algorithm are marked in <...>.
    Time reported is for the clique search only (reading file from disk and transforming the graph into adjacency matrix are not counted)
    
