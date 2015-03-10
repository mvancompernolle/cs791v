#ifndef MC_H
#define MC_H

#include <vector>
#include <sys/time.h>
#include <time.h>

class MC{
public:
	MC(int n, std::vector<std::vector<int> > A, std::vector<int> degree);
	virtual ~MC();
	virtual void search();
	virtual void setTimeLimit(long t);
	virtual int getMaxSize();
	virtual int getNumNodes();
	virtual std::vector<int> getSolution();

protected:
	std::vector<int> degree;
	std::vector<std::vector<int> > A;
	int n;
	long nodes;
	long timeLimit;
	int maxSize;
	int style;
	std::vector<int> solution;
	timeval tod1, tod2;
	long long int todiff(struct timeval *tod1, struct timeval *tod2);
	void expand(std::vector<int> C, std::vector<int> P);
	virtual void saveSolution(std::vector<int> C);
	void printVector(const std::vector<int>& v) const;
	void printArray(const int a[], int size) const;
};

#endif