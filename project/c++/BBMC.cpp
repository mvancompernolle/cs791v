#include "BBMC.h"
#include <iostream>

BBMC::BBMC(int n, std::vector<std::vector<int> > A, std::vector<int> degree, int style) : MCQ(n, A, degree, style){
	// N stores the neighborhood of vertex v
	N = new bitset[n];
	// invN stores the inverse neighborhood of vertex v
	invN = new bitset[n];
	V.resize(n);
}

BBMC::~BBMC(){
}

void BBMC::search(){
	gettimeofday(&tod1, NULL);
	nodes = 0;

	// current clique encoded as a bit string
	bitset C(n);
	// candidate set encoded as a bit string
	bitset P(n);

	for(int i=0; i<n; i++){
		N[i].bits.resize(n);
		invN[i].bits.resize(n);
		V[i].index = i;
		V[i].degree = degree[i];

	}

	// order vertices
	orderVertices();
	for(int i=0; i<C.bits.size(); i++){
		C.bits[i] = 0;
		P.bits[i] = 1;
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
			N[i].bits[j] = (A[u][v] == 1);
			invN[i].bits[j] = (A[u][v] == 0);
		}
	}

}

void BBMC::BBMaxClique(bitset C, bitset P){
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

	int m = calcCardinality(P.bits);
	int U[m];
	int color[m];
//std::cout << m << std::endl;
	BBColor(P, U, color);
//printArray(U, m);
//printArray(color, m);
	// iterate over the candidate set
	for(int i=m-1; i>= 0; i--){

		// return if clique cannot grow large enough to be maximum clique
		if(color[i] + calcCardinality(C.bits) <= maxSize) 
			return;

		// select a vertex from P and add it to the current clique
		bitset newP(P);
		int v = U[i];
		C.bits[v] = 1;

		// perform bitwise and
		for(int i=0; i<newP.bits.size(); i++){
			newP.bits[i] = newP.bits[i] * N[v].bits[i];
		}
/*
std::cout << "N: ";
printBitSet(N[v]);
std::cout << "newP: ";
printBitSet(newP);
*/
		// if newP is empty is is maximal, so stop searching and save it if it is maximum
		if(newP.bits.none() && calcCardinality(C.bits) > maxSize){
//std::cout << "saved" << std::endl;
			saveSolution(C);
		}
		// else recursively continue search 
		else if(!newP.bits.none()){
//std::cout << "called again" << std::endl;
			BBMaxClique(C, newP);
		}

		// remove v from P and C when returning
		C.bits[v] = 0;
		P.bits[v] = 0;

		//gettimeofday(&t2, NULL);
		//std::cout << todiff(&t2, &t1)/1000 << std::endl;
	}

}

void BBMC::BBColor(bitset P, int U[], int color[]){
	bitset copyP(P);
	int v;
	int colorClass = 0;
	int i = 0;
	int card = 0, qCard = 0;
	card = calcCardinality(copyP.bits);

	while(card != 0){
		colorClass++;
		bitset Q(copyP);

		qCard = calcCardinality(Q.bits);

		while(qCard != 0){
			// return the index of the first set bit
			for(int i=0; i<Q.bits.size(); i++){
				if(Q.bits[i]){
					v = i;
					break;
				}
			}
			copyP.bits[v] = 0;
			Q.bits[v] = 0;
			// perform a bitwise and operation
			for(int i=0; i<Q.bits.size(); i++){
				Q.bits[i] = Q.bits[i] * invN[v].bits[i];
			}
			U[i] = v;
			color[i++] = colorClass;

			qCard = calcCardinality(Q.bits);
		}

		card = calcCardinality(copyP.bits);
	}
}

void BBMC::saveSolution(bitset C){
	std::fill(solution.begin(), solution.end(), 0);
	for(int i=0; i<C.bits.size(); i++){
		if(C.bits[i])
			solution[V[i].index] = 1;
	}
	maxSize = calcCardinality(C.bits);
}

int BBMC::calcCardinality(const boost::dynamic_bitset<>& bitSet) const{
	int count = 0;
	for(int i=0; i<bitSet.size(); i++)
		count += bitSet[i];
	return count;
}

void BBMC::printBitSet(const boost::dynamic_bitset<>& bitSet) const{
	for(int i=0; i<bitSet.size(); i++){
		std::cout << bitSet[i] << " ";
	}
	std::cout << std::endl;
}