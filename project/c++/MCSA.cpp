#include "MCSA.h"
#include <iostream>
#include <algorithm>
#include <stack>

MCSA::MCSA(int n, std::vector<std::vector<int> > A, std::vector<int> degree, int style): MCQ(n, A, degree, style){

}

MCSA::~MCSA(){

}

// find the largest clique or terminates after time limit
void MCSA::search(){
	gettimeofday(&tod1, NULL);
	nodes = 0;
	// contains vertices i+1 and used when sorting vertices by their color
	colorClass.resize(n);
	// current clique found
	std::vector<int> C;
	C.reserve(n);
	// vertices that may be added to growing clique (candidate set)
	std::vector<int> P(n);
	std::vector<int> colOrd(n);
	// init to have all vertices as possible candidates
	for(int i=0; i<n; i++)
		colorClass[i].resize(n);

	// order vertices
	orderVertices(colOrd);
/*
for(int& p: P)
	std::cout << p << " ";
std::cout << std::endl;
*/
	expand(C, P, colOrd);
}

void MCSA::expand(std::vector<int> C, std::vector<int> P, std::vector<int> colOrd){
	int w, u;

	// see if the time limit has been reached
	gettimeofday(&tod2, NULL);
	if(timeLimit > 0 && todiff(&tod2, &tod1)/1000 >= timeLimit) 
		return;

	// count the size of the backtrack search tree explored
	nodes++;

	int m = colOrd.size();
	int color[m];
	numberSort(C, colOrd, P, color);
/*
std::cout << "Color: ";
for(int i=0; i<m; i++){
	std::cout << P[i] << " ";
}
std::cout << std::endl;
*/
	// iterate over the candidate set
	for(int i=P.size()-1; i>= 0; i--){

//std::cout << "Loop:" << i << " Max: " << maxSize << std::endl;

		//timeval t1, t2;
		//gettimeofday(&t1, NULL);
/*
// print C and P
std::cout << "C: ";
for(auto& w: C){
	std::cout << w << " ";
}
std::cout << "\t";
std::cout << "P: ";
for(auto& w: P){
	std::cout << w << " ";
}
std::cout << std::endl;
*/
		// return if clique cannot grow large enough to be maximum clique
		if(C.size() + color[i] <= maxSize) 
			return;

		// select a vertex from P and add it to the current clique
		int v = P[i];
		C.push_back(v);

//std::cout << "V: " << v << std::endl;

		// newP is the set of vertices in P that are adjacent to vertex v
		// all vertices in newP are adjacent to all vertices in C and all pairs of vertices in C ar adjacent
		std::vector<int> newP;
		std::vector<int> newColOrd;
		newP.reserve(i);
		newColOrd.reserve(i);
		for(int j=0; j<=i; j++){
			u = P[j];
			if(A[u][v] == 1){
				newP.push_back(w);
			}
			w = colOrd[j];
			if(A[v][w] == 1){
				newColOrd.push_back(w);
			}
		}

		// if newP is empty is is maximal, so stop searching and save it if it is maximum
		if(newP.empty() && C.size() > maxSize){
//std::cout << "Saving" << std::endl;
			saveSolution(C);
		}
		// else recursively continue search 
		else if(!newP.empty()){
//std::cout << "Expanding" << std::endl;
			expand(C, newP, newColOrd);
		}

		// remove v from P and C when returning
		C.pop_back();
		P.pop_back();
		colOrd.erase(std::remove(colOrd.begin(), colOrd.end(), v), colOrd.end());
		//gettimeofday(&t2, NULL);
		//std::cout << todiff(&t2, &t1)/1000 << std::endl;
	}
}