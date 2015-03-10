#ifndef BBMC_H
#define BBMC_H

#include <boost/dynamic_bitset.hpp>
#include <vector>
#include "Vertex.h"
#include "MCQ.h"

class BBMC : public MCQ{
public:
	BBMC(int n, std::vector<std::vector<int> > A, std::vector<int> degree, int style);
	~BBMC();
	void search();
	void orderVertices();
	void BBMaxClique(boost::dynamic_bitset<> C, boost::dynamic_bitset<> P);
	void BBColor(boost::dynamic_bitset<> P, int U[], int color[]);
	void saveSolution(boost::dynamic_bitset<> C);

private:
	boost::dynamic_bitset<>* N;
	boost::dynamic_bitset<>* invN;
	std::vector<Vertex> V;
	int calcCardinality(const boost::dynamic_bitset<>& bitset) const;
	void printBitSet(const boost::dynamic_bitset<>& bitset) const;
};

#endif