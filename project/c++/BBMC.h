#ifndef BBMC_H
#define BBMC_H

#include <boost/dynamic_bitset.hpp>
#include <vector>
#include "Vertex.h"
#include "MCQ.h"

struct bitset{
	boost::dynamic_bitset<> bits;
	int cardinality;

	bitset(){
		cardinality = 0;
	}

	bitset(int n){
		bits.resize(n);
		cardinality = 0;
	}
};

class BBMC : public MCQ{
public:
	BBMC(int n, std::vector<std::vector<int> > A, std::vector<int> degree, int style);
	~BBMC();
	void search();
	void orderVertices();
	void BBMaxClique(bitset C, bitset P);
	void BBColor(bitset P, int U[], int color[]);
	void saveSolution(bitset C);

private:
	bitset* N;
	bitset* invN;
	std::vector<Vertex> V;
	int calcCardinality(const boost::dynamic_bitset<>& bitset) const;
	void printBitSet(const boost::dynamic_bitset<>& bitset) const;
};

#endif