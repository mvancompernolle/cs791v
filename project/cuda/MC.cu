#include "MC.h"
#include <iostream>
#include <algorithm>
#include "cuda.h"
#include <queue>

MC::MC(int n, std::vector<std::vector<int> > A, std::vector<int> degree){
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

	/*// allocate memory on GPU for adjacency matrix
	int *dev_adj;
		cudaError_t err = cudaMalloc( (void**) &dev_adj, A[0].size() * sizeof(int));
		if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	
	cudaMemcpy(dev_adj, A[0].data(), A[0].size() * sizeof(int), cudaMemcpyHostToDevice);*/
}

MC::~MC(){

}

// find the largest clique or terminates after time limit
void MC::search(){
	gettimeofday(&tod1, NULL);
	nodes = 0;
	// current clique found
	std::vector<int> C;
	// vertices that may be added to growing clique (candidate set)
	std::vector<int> P(n);
	// init to have all vertices as possible candidates
	for(int i=0; i<n; i++)
		P[i] = i;
	expand(C, P);
}

void MC::expand(std::vector<int> C, std::vector<int> P){
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

void MC::saveSolution(std::vector<int> C){
	std::fill(solution.begin(), solution.end(), 0);
	for(auto& i : C)
		solution[i] = 1;
	maxSize = C.size();
}

void MC::setTimeLimit(long t){
	timeLimit = t;
}

int MC::getMaxSize(){
	return maxSize;
}

int MC::getNumNodes(){
 	return nodes;
}

std::vector<int> MC::getSolution(){
	return solution;
}

long long int MC::todiff(struct timeval *tod1, struct timeval *tod2)
{
	long long t1, t2;
	t1 = tod1->tv_sec * 1000000 + tod1->tv_usec;
	t2 = tod2->tv_sec * 1000000 + tod2->tv_usec;
	return t1 - t2;
}

void MC::printVector(const std::vector<int>& v) const{
	for(int i=0; i<v.size(); i++)
		std::cout << v[i] << " ";
	std::cout << std::endl;
}

void MC::printArray(const int a[], int size) const{
	for(int i=0; i<size; i++)
		std::cout << a[i] << " ";
	std::cout << std::endl;
}
