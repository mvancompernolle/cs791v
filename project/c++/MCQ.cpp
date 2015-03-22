#include "MCQ.h"
#include <iostream>
#include <algorithm>
#include <stack>

MCQ::MCQ(int n, std::vector<std::vector<int> > A, std::vector<int> degree, int style): MC(n, A, degree){
	// number of nodes in the graph
	this->n = n;
	// equals 1 if an edge exists
	this->A = A;
	// the number of vertices adjacent to vertex i
	this->degree = degree;
/*
for(int& i: degree) std::cout << i << " ";
std::cout << std::endl;
*/
	// largest clique found so far
	nodes = maxSize = 0;
	timeLimit = -1;
	// flag to customise the algorithm with respect to ordering of the vertices
	this->style = style;
	// largest clique found
	solution.resize(n);
}

MCQ::~MCQ(){

}

// find the largest clique or terminates after time limit
void MCQ::search(){
	gettimeofday(&tod1, NULL);
	nodes = 0;
	// contains vertices i+1 and used when sorting vertices by their color
	colorClass.resize(n);
	// current clique found
	std::vector<int> C;
	C.reserve(n);
	// vertices that may be added to growing clique (candidate set)
	std::vector<int> P(n);
	// init to have all vertices as possible candidates
	for(int i=0; i<n; i++)
		colorClass[i].resize(n);

	// order vertices
	orderVertices(P);
/*
for(int& p: P)
	std::cout << p << " ";
std::cout << std::endl;
*/
	expand(C, P);
}

void MCQ::expand(std::vector<int> C, std::vector<int> P){
	int w;

	// see if the time limit has been reached
	gettimeofday(&tod2, NULL);
	if(timeLimit > 0 && todiff(&tod2, &tod1)/1000 >= timeLimit) 
		return;

	// count the size of the backtrack search tree explored
	nodes++;

	int m = P.size();
	int color[m];
	numberSort(C, P, P, color);
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
		newP.reserve(i);
		for(int j=0; j<=i; j++){
			w = P[j];
			if(A[w][v] == 1){
				newP.push_back(w);
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
			expand(C, newP);
		}

		// remove v from P and C when returning
		C.pop_back();
		P.pop_back();

		//gettimeofday(&t2, NULL);
		//std::cout << todiff(&t2, &t1)/1000 << std::endl;
	}
}

void MCQ::orderVertices(std::vector<int>& verts){
	// create the vertices to sort
	std::vector<Vertex> V(n);
	for(int i=0; i<n; i++){
		V[i].index = i;
		V[i].degree = degree[i];
	}

	// calculate the sum of the neighboring degrees
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			if(A[i][j] == 1)
				V[i].setNebDeg(V[i].getNebDeg() + degree[j]);
		}
	}
/*
std::cout << "NebDeg: ";
for(Vertex& v: V) std::cout << v.getNebDeg() << " ";
std::cout << std::endl;
*/
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

	for(int i=0; i<V.size(); i++){
		verts[i] = V[i].index;
	}
}

bool MCQ::conflicts(int v, std::vector<int> cClass){
	// return true if pass in vertex is adjacent to any of the vertices in color class
	for(int i=0; i<cClass.size(); i++){
		int w = cClass[i];
		if(A[v][w] == 1)
			return true;
	}
	// return false otherwise
	return false;
}

void MCQ::numberSort(std::vector<int> C, std::vector<int> colOrd, std::vector<int>& P, int color[]){
	// records the number of colors used
	int colors = 0;
	int m = colOrd.size();

	// clear out the color classes that might be used
	for(int i=0; i<m; i++){
		colorClass[i].clear();
	}

	// vertices are selected from colOrd and placed into first color class in which there are no conflicts
	for(int i=0; i<m; i++){
		int v = colOrd[i];
		int k = 0;
		// vertice in colorClass are not pair wise adjacent and have same color (independent set)
		while (conflicts(v, colorClass[k]))
			k++;
		colorClass[k].push_back(v);
		colors = std::max(colors, k+1);
	}
/*
std::cout << "ColorClasses: " << std::endl;
for(int i=0; i<colors; i++){
	std::cout << "C" << i << " ";
	for(int j=0; j<colorClass[i].size(); j++){
		std::cout << colorClass[i][j] << " ";
	}
	std::cout << std::endl;
}
*/
	// pidgeon hole sort
	P.clear();
	int i = 0;
	for(int k=0; k<colors; k++){
		for(int j=0; j<colorClass[k].size(); j++){
			int v = colorClass[k][j];
			P.push_back(v);
			color[i++] = k+1;
		}
	}

/*
std::cout << "Pidgeon Hole: ";
for(int& v: P) std::cout << v << " ";
std::cout << std::endl;

std::cout << "Colors: ";
for(int i=0; i<m; i++) std::cout << color[i] << " ";
std::cout << std::endl;
*/
}

void MCQ::minWidthOrder(std::vector<Vertex>& V){
	std::vector<Vertex> L;
	std::stack<Vertex> S;

	for(int i=0; i<V.size(); i++){
		L.push_back(V[i]);
	}

	while(!L.empty()){
		// select vertex with smallest degree and store in v
		int pos = 0;
		Vertex v = L[0];
		for(int i=0; i<L.size(); i++){
			if(L[i].degree < v.degree){
				v = L[i];
				pos = i;
			}
		}

		// push v onto stack and remove from L
		S.push(v);
		L.erase(L.begin()+pos);

		// reduce degree of all vertices in L that are adjacent to v
		for(Vertex& u : L){
			if(A[u.index][v.index] == 1)
				u.degree--;
		}
	}

	// pop of stack and place onto V giving minimum width ordering
	int k = 0;
	while(!S.empty()){
		V[k++] = S.top();
		S.pop();
	}
}
