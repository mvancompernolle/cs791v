#include <iostream>
#include "pablodev/utils/benchmark.h"

using namespace std;

class BkClique:public Benchmark{
public:
	BkClique(string path):Benchmark(path){}

	void Dimacs();
	void SubsetDimacs();
	void HardDimacs();
	void ILS();
	void EasyILS();
	void SubsetBhoshlib();
	void Snap();
	void Others();					//for general purpose experiments
};