#ifndef BBMC_H
#define BBMC_H

#include <boost/dynamic_bitset.hpp>
#include <vector>
#include "Vertex.h"
#include "MCQ.h"
#include <queue>
#include <mutex>

struct workItem{
	boost::dynamic_bitset<> C, P;
};

class BBMC : public MCQ{
public:
	BBMC(int n, std::vector<std::vector<int> > A, std::vector<int> degree, int style);
	~BBMC();
	void search();
	void orderVertices();
	void BBMaxClique(boost::dynamic_bitset<> C, boost::dynamic_bitset<> P, int tid);
	void runWorker(int tid);
	void BBColor(const boost::dynamic_bitset<>& P, int U[], int color[]);
	void saveSolution(const boost::dynamic_bitset<>& C);
	void printQueue();

private:
	boost::dynamic_bitset<>* N;
	boost::dynamic_bitset<>* invN;
	std::vector<Vertex> V;
	std::queue<workItem> work;
	int popThread;
	int numIdle;
	void printBitSet(const boost::dynamic_bitset<>& bitset) const;
	std::mutex popMutex, idleMutex;
};

#endif