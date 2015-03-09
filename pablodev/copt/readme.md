Combinatorial Optimization Algorithms 
===================

This block currently contains an implementation of state-of-the-art BBMCI, an exact algorithm for the **maximum clique problem (MCP)**, as well as several utilities necessary for MCP such as different vertex orderings by degree or greedy sequential independent set coloring to compute bounds.

BBMCI is described in [1-2] and several applications of BBMCS are reported in literature connected to the Maximum Common Subgraph problem, as in [3-4]. It is currently being actively developped and several improvements have been proposed such as [5]. The algorithm uses several improvements over previously known algorithms based on a bitstring encoding. This implementation uses the [BITSCAN](https://www.biicode.com/pablodev/bitscan) library for bitstring and [GRAPH](https://www.biicode.com/pablodev/graph) library for graphs. Both libraries are available in the Biicode repository.

The block also includes current research in large scale networks by extending BBMCI to a sparse bitstring model again using BITSCAN. It also includes a parallel version of BBMCI using [OpenMP](http://openmp.org/wp/) framework. Please note that at the moment only BBMCI is fully operative. The remainder of the soure code will be released as soon as results are published (nevertheless a big part of the latter source code is available).

The source code has been tested on **Windows** and **Linux** platforms. 

Performance of this implementation is similar to that described in the original papers. The developer is also the author of BBMCI.


Terms and conditions
-------------------------------

Please feel free to use this code. *Just remember it is a research prototype and may not work for you*.The only condition is that you cite references [1-2] in your work.

Finally your feedback is gladly welcome. If you run into any problems just send your specific question to Biicode forum or, if its a general issue, to *StackOverflow* (tags Biicode, C++).

References
-------------------------------
[1] *[An exact bit-parallel algorithm for the maximum clique problem](http://dl.acm.org/citation.cfm?id=1860369%20)*.

[2] *[An improved bit parallel exact maximum clique algorithm](http://link.springer.com/article/10.1007%2Fs11590-011-0431-y)*.

[3] *[Robust Global Feature Based Data Association With a Sparse Bit Optimized Maximum Clique Algorithm](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6527958).*

[4] *[Fast exact feature based data correspondence search with an efficient bit-parallel MCP solver](http://http://link.springer.com/article/10.1007%2Fs10489-008-0147-6)*.

[5] *[Relaxed approximate coloring in exact maximum clique search](http://dl.acm.org/citation.cfm?id=2566230)*.


   