#ifndef MCQ_H
#define MCQ_H

#include <vector>
#include <sys/time.h>
#include <time.h>
#include "Vertex.h"
#include "MC.h"

class MCQ : public MC{
public:
	MCQ(int n, std::vector<std::vector<int> > A, std::vector<int> degree, int style);
	virtual ~MCQ();
	virtual void search();
	virtual void expand(std::vector<int> C, std::vector<int> P);

protected:
	virtual void orderVertices(std::vector<int>& verts);
	virtual void numberSort(std::vector<int> C, std::vector<int> colOrd, std::vector<int>& P, int color[]);
	virtual void minWidthOrder(std::vector<Vertex>& V);
	virtual bool conflicts(int v, std::vector<int> cClass);
	std::vector<std::vector<int> > colorClass;
};

#endif