#ifndef BBMC_H
#define BBMC_H

#include <boost/dynamic_bitset.hpp>
#include <vector>
#include "Vertex.h"
#include "MCQ.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


class BBMC : public MCQ{
public:
	BBMC(int n, std::vector<std::vector<int> > A, std::vector<int> degree, int style);
	~BBMC();
	void searchParallel();
	void generateInitialNodes();
	void orderVertices();
	void BBColor(const boost::dynamic_bitset<>& P, int U[], int color[]);

private:
	boost::dynamic_bitset<>* N;
	boost::dynamic_bitset<>* invN;
	std::vector<Vertex> V;
	int numInts;
	thrust::host_vector<unsigned int> activeC;
	thrust::host_vector<unsigned int> activeP;
	void printBitSet(const boost::dynamic_bitset<>& bitset) const;
	void printIntArray(unsigned int* arr, int n, int numInts) const;

};

#endif