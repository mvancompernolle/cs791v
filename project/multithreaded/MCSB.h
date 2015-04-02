#ifndef MCSB_H
#define MCSB_H

#include <vector>
#include <sys/time.h>
#include <time.h>
#include "MCSA.h"

// uses an adjunct ordered set of verties for approximate coloring
// uses a repair mechanism when coloring vertices

class MCSB : public MCSA{
public:
	MCSB(int n, std::vector<std::vector<int> > A, std::vector<int> degree, int style);
	virtual ~MCSB();
	virtual void numberSort(std::vector<int> C, std::vector<int> colOrd, std::vector<int>& P, int color[]);
	bool repair(int v, int k);
	int getSingleConflictVariable(int v, std::vector<int> cClass);
private:
};

#endif