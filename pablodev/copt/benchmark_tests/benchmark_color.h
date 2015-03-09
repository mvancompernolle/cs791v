#include <iostream>
#include "pablodev/utils/benchmark.h"

using namespace std;

class BkColor:public Benchmark{
public:
	BkColor(string path):Benchmark(path){}

	void Dimacs();
	void MinSumEasy();
	void MinSumHard();
	void Mug();
	void Wap();
	void Insertions();
	void Init();
	void Zeroin();
	void Dsjc();
	void Le4();
	void Flat();
	void DimacsPreProcessable();
	void EasyDimacs();
	void NonTrivialMalagutiDimacs();
	void MyRandomBenchmark();				
};