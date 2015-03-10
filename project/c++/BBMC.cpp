#include "BBMC.h"
#include <iostream>

BBMC::BBMC(int n, std::vector<std::vector<int> > A, std::vector<int> degree, int style) : MCQ(n, A, degree, style){
	// N stores the neighborhood of vertex v
	N = new boost::dynamic_bitset<>[n];
	// invN stores the inverse neighborhood of vertex v
	invN = new boost::dynamic_bitset<>[n];
	V.resize(n);
}

BBMC::~BBMC(){
}

void BBMC::search(){
	gettimeofday(&tod1, NULL);
	nodes = 0;

	// current clique encoded as a bit string
	boost::dynamic_bitset<> C(n);
	// candidate set encoded as a bit string
	boost::dynamic_bitset<> P(n);

	for(int i=0; i<n; i++){
		N[i].resize(n);
		invN[i].resize(n);
		V[i].index = i;
		V[i].degree = degree[i];

	}

	// order vertices
	orderVertices();
	for(int i=0; i<C.size(); i++){
		C[i] = 0;
		P[i] = 1;
	}
/*
printBitSet(C);
printBitSet(P);
for(int i=0; i<n; i++)
	printBitSet(invN[i]);
for(Vertex& v: V){
	std::cout << v.index << "-" << v.degree << " ";
}
std::cout << std::endl;
*/
	BBMaxClique(C, P);

/*
for(int& p: P)
	std::cout << p << " ";
std::cout << std::endl;
*/
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
			N[i][j] = (A[u][v] == 1);
			invN[i][j] = (A[u][v] == 0);
		}
	}

}

void BBMC::BBMaxClique(boost::dynamic_bitset<> C, boost::dynamic_bitset<> P){
	int w;
/*
std::cout << std::endl;
printBitSet(C);
printBitSet(P);
*/
	// see if the time limit has been reached
	gettimeofday(&tod2, NULL);
	if(timeLimit > 0 && todiff(&tod2, &tod1)/1000 >= timeLimit) 
		return;

	// count the size of the backtrack search tree explored
	nodes++;

	int m = calcCardinality(P);
	int U[m];
	int color[m];
//std::cout << m << std::endl;
	BBColor(P, U, color);
//printArray(U, m);
//printArray(color, m);
	// iterate over the candidate set
	for(int i=m-1; i>= 0; i--){

		// return if clique cannot grow large enough to be maximum clique
		if(color[i] + calcCardinality(C) <= maxSize) 
			return;

		// select a vertex from P and add it to the current clique
		boost::dynamic_bitset<> newP(P);
		int v = U[i];
		C[v] = 1;

		// perform bitwise and
		for(int i=0; i<newP.size(); i++){
			newP[i] = newP[i] * N[v][i];
		}
/*
std::cout << "N: ";
printBitSet(N[v]);
std::cout << "newP: ";
printBitSet(newP);
*/
		// if newP is empty is is maximal, so stop searching and save it if it is maximum
		if(newP.none() && calcCardinality(C) > maxSize){
//std::cout << "saved" << std::endl;
			saveSolution(C);
		}
		// else recursively continue search 
		else if(!newP.none()){
//std::cout << "called again" << std::endl;
			BBMaxClique(C, newP);
		}

		// remove v from P and C when returning
		C[v] = 0;
		P[v] = 0;

		//gettimeofday(&t2, NULL);
		//std::cout << todiff(&t2, &t1)/1000 << std::endl;
	}

}

void BBMC::BBColor(boost::dynamic_bitset<> P, int U[], int color[]){
	boost::dynamic_bitset<> copyP(P);
	int v;
	int colorClass = 0;
	int i = 0;
	int card = 0, qCard = 0;
	card = calcCardinality(copyP);

	while(card != 0){
		colorClass++;
		boost::dynamic_bitset<> Q(copyP);

		qCard = calcCardinality(Q);

		while(qCard != 0){
			// return the index of the first set bit
			for(int i=0; i<Q.size(); i++){
				if(Q[i]){
					v = i;
					break;
				}
			}
			copyP[v] = 0;
			Q[v] = 0;
			// perform a bitwise and operation
			for(int i=0; i<Q.size(); i++){
				Q[i] = Q[i] * invN[v][i];
			}
			U[i] = v;
			color[i++] = colorClass;

			qCard = calcCardinality(Q);
		}

		card = calcCardinality(copyP);
	}
}

void BBMC::saveSolution(boost::dynamic_bitset<> C){
	std::fill(solution.begin(), solution.end(), 0);
	for(int i=0; i<C.size(); i++){
		if(C[i])
			solution[V[i].index] = 1;
	}
	maxSize = calcCardinality(C);
}

int BBMC::calcCardinality(const boost::dynamic_bitset<>& bitset) const{
	int count = 0;
	for(int i=0; i<bitset.size(); i++)
		count += bitset[i];
	return count;
}

void BBMC::printBitSet(const boost::dynamic_bitset<>& bitset) const{
	for(int i=0; i<bitset.size(); i++){
		std::cout << bitset[i] << " ";
	}
	std::cout << std::endl;
}