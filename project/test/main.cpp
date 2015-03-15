/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 */

#include <iostream>
#include <bitset>
#include <cassert>

#define TRACE_LEVEL 1
#define TRACE_MASK TRACE_MASK_CLIQUE

#include "Dimacs.h"
#include "MaximumCliqueBase.h"
#include "ParallelMaximumClique.h"
#include "BB_GreedyColorSort.h"
#include "BB_ColorRSort.h"
#include "DegreeSort.h"
#include "McrBB.h"


void pBbmcTest(const char* params, int numThreads, int numJobs, std::vector<int>& affinities) {
    std::string testCliqueFile = ( params == nullptr ? "../clq/brock200_1.clq" : params);

    // DIMACS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    std::cout << "Loading DIMACS graph " << testCliqueFile << "\n";
    Dimacs dimacsGraph;
    bool graphLoaded = dimacsGraph.load(testCliqueFile.c_str());
    if (!graphLoaded)
        throw "Unable to load graph\n";
    
    bool coloredOut = false;
#ifdef __linux
    coloredOut = true;
#endif

    Graph<BitstringSet> graphB;
    graphB.init(dimacsGraph.getAdjacencyMatrix(), dimacsGraph.getDegrees());
    std::cout << dimacsGraph.getNumVertices() << " vertices " << dimacsGraph.getNumEdges() << " edges " 
        << dimacsGraph.getNumEdges()*2.0/(dimacsGraph.getNumVertices()*(dimacsGraph.getNumVertices()-1)) << "\n";
    
    if (numThreads == 0) {        
        MaximumCliqueProblem<
            int,                        // vertex ID
            BitstringSet,               // vertex set
            Graph<BitstringSet>,        // graph
            BBMcrSort<BBColorRSort<Graph<BitstringSet>>>,        // color sort
            DegreeSort
            > problem(graphB);
        problem.search();
        problem.outputStatistics(coloredOut); std::cout << "\n\n"; 
    } else {
        ParallelMaximumCliqueProblem<
            int,                        // vertex ID
            BitstringSet,               // vertex set
            Graph<BitstringSet>,        // graph
            BBMcrSort<BBColorRSort<Graph<BitstringSet>>>,        
            DegreeSort
            > problemP1(graphB);
        problemP1.search(numThreads, numJobs, affinities);
        problemP1.outputStatistics(coloredOut); std::cout << "\n"; 
        std::cout << "Thread efficiency = " << std::setprecision(3) << problemP1.workerEfficiency() << "\n\n";
    }
}


int main(int argc, char** argv) {
    try {
        std::vector<int> bindProcessors;
        int numThreads = 0, numJobs = 0;
        if (argc > 2) {
            numThreads = atoi(argv[2]);
            if (numThreads < 0)
                numThreads = 0;
            if (argc > 3) {
                numJobs = atoi(argv[3]);
                if (numJobs < 0)
                    numJobs = 0;
                if (argc > 4) {
                    for (int argi = 4; argi < argc; ++argi)
                        bindProcessors.push_back(atoi(argv[argi]));
                }
            }
        }
        pBbmcTest(argc > 1 ? argv[1] : nullptr, numThreads, numJobs, bindProcessors);
    } catch (const char* e) {
        std::cout << "Terminated due to exception: " << e << std::endl;
    } catch (...) {
        std::cout << "Terminated due to unknown exception: " << std::endl;
    }
    return 0;
}
