#ifndef MCSA_H
#define MCSA_H

#include <vector>
#include <sys/time.h>
#include <time.h>
#include "MCQ.h"

// uses an adjunct ordered set of verties for approximate coloring
// uses a repair mechanism when coloring vertices

class MCSA : public MCQ{
public:
	MCSA(int n, std::vector<std::vector<int> > A, std::vector<int> degree, int style);
	virtual ~MCSA();
	virtual void search();
	virtual void expand(std::vector<int> C, std::vector<int> P, std::vector<int> colOrd);

private:
};

#endif