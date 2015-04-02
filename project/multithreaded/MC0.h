#ifndef MC0_H
#define MC0_H

#include <vector>
#include <sys/time.h>
#include <time.h>
#include "MC.h"

class MC0 : public MC{
public:
	MC0(int n, std::vector<std::vector<int> > A, std::vector<int> degree);
	virtual ~MC0();
	virtual void expand(std::vector<int> C, std::vector<int> P);

private:
};

#endif