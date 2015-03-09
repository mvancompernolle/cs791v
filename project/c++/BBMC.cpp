#include "BBMC.h"

BBMC::BBMC(int n, std::vector<std::vector<int> > A, std::vector<int> degree, int style) : MCQ(n, A, degree, style){
	N = new boost::dynamic_bitset<>(n);
	invN = new boost::dynamic_bitset<>(n);
	V.resize(n);
}

BBMC::~BBMC(){
	gettimeofday(&tod1, NULL);
	nodes = 0;

	boost::dynamic_bitset<> C(n);
	boost::dynamic_bitset<> P(n);

	for(int i=0; i<n; i++){
		N[i].resize(n);
		N[i].resize(n);
		V[i].index = i;
		V[i].degree = degree[i];
	}

	// order vertices
	orderVertices();
	for(int i=0; i<C.size(); i++){
		C[i] = 0;
		P[i] = 0;
	}

	BBMaxClique(C, P);
/*
for(int& p: P)
	std::cout << p << " ";
std::cout << std::endl;
*/
}

void BBMC::search(){
}

void BBMC::orderVertices(){
	// calculate the sum of the neighboring degrees
	for(int i=0; i<n; i++){
		V[i].index = i;
		V[i].degree = degree[i];

		for(int j=0; j<n; j++){
			if(A[i][j] == 1)
				V[i].setNebDeg(V[i].getNebDeg() + degree[j]);
		}
	}

	// order based on style passed in
	switch(style){
		case 1:
			// order by non-increasing degree, tie-breaking on index
			std::sort(V.begin(), V.end(), Vertex::VertexCmp);
			break;
		case 2:
			// order by minimum width order
			minWidthOrder(V);
			break;
		case 3:
			// known as MCR
			// order by non-increasing degree, tie-greaking on sum of the neighborhood
			// degree nubDeg and then on index
			std::sort(V.begin(), V.end(), Vertex::MCRComparator);
			break;
	}

	int u, v;
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			u = V[i].index;
			v = V[j].index;
			invN[i][j] = (A[u][v] == 0);
		}
	}

}

void BBMC::BBMaxClique(boost::dynamic_bitset<> C, boost::dynamic_bitset<> P){
	int w;

	// see if the time limit has been reached
	gettimeofday(&tod2, NULL);
	if(timeLimit > 0 && todiff(&tod2, &tod1)/1000 >= timeLimit) 
		return;

	// count the size of the backtrack search tree explored
	nodes++;

	int m = 0;
	for(int i=0; i<P.size(); i++)
		m += P[i];

	int U[m];
	int color[m];
	BBColor(P, U, color);
/*
	// iterate over the candidate set
	for(int i=P.size()-1; i>= 0; i--){

		// return if clique cannot grow large enough to be maximum clique
		if(C.size() + color[i] <= maxSize) 
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
			if(A[w][v] == 1){
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

		//gettimeofday(&t2, NULL);
		//std::cout << todiff(&t2, &t1)/1000 << std::endl;
	}
*/
}

void BBMC::BBColor(boost::dynamic_bitset<> P, int U[], int color[]){
	boost::dynamic_bitset<> copyP(P);
	int colorClass = 0;
	int i = 0;
	int card = 0;
	for(int i=0; i<copyP.size(); i++){
		card += copyP[i];
	}
	/*while(){

	}*/
}

void BBMC::saveSolution(boost::dynamic_bitset<> C){
	std::fill(solution.begin(), solution.end(), 0);
	for(int i=0; i<C.size(); i++){
		if(C[i])
			solution[V[i].index] = 1;
	}
	maxSize = 0;
	for(int i=0; i<C.size(); i++){
		maxSize += C[i];
	}
}