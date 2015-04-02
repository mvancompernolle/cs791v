#include "MC0.h"
#include <iostream>
#include <algorithm>

MC0::MC0(int n, std::vector<std::vector<int> > A, std::vector<int> degree) : MC(n, A, degree){
	// number of nodes in the graph
	this->n = n;
	// equals 1 if an edge exists
	this->A = A;
	// the number of vertices adjacent to vertex i
	this->degree = degree;
	// largest clique found so far
	nodes = maxSize = 0;
	timeLimit = -1;
	// flag to customise the algorithm with respect to ordering of the vertices
	style = 1;
	// largest clique found
	solution.resize(n);
}

MC0::~MC0(){

}

void MC0::expand(std::vector<int> C, std::vector<int> P){
	int w;
	/*// print C and P
	std::cout << "C: ";
	for(auto& w: C){
		std::cout << w << " ";
	}
	std::cout << "\t";
	std::cout << "P: ";
	for(auto& w: P){
		std::cout << w << " ";
	}
	std::cout << std::endl;*/

	// see if the time limit has been reached
	gettimeofday(&tod2, NULL);
	if(timeLimit > 0 && todiff(&tod2, &tod1)/1000 >= timeLimit) 
		return;

	// count the size of the backtrack search tree explored
	nodes++;

	// iterate over the candidate set
	for(int i=P.size()-1; i>= 0; i--){

		// return if clique cannot grow large enough to be maximum clique
		if(C.size() + P.size() <= maxSize) 
			return;

		// select a vertex from P and add it to the current clique
		int v = P[i];
		C.push_back(v);

		// newP is the set of vertices in P that are adjacent to vertex v
		// all vertices in newP are adjacent to all vertices in C and all pairs of vertices in C ar adjacent
		std::vector<int> newP;
		newP.reserve(i);
		for(int j=0; j<=i; j++){
			w = P[j];
			if(A[v][w] == 1){
				newP.push_back(w);
			}
		}

		// if newP is empty is is maximal, so stop searching and save it if it is maximum
		if(newP.empty() && C.size() > maxSize){
			saveSolution(C);
		}
		// else recursively continue search 
		else if(!newP.empty()){
			expand(C, newP);
		}

		// remove v from P and C when returning
		C.pop_back();
		P.pop_back();
	}
}
