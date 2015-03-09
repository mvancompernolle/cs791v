#ifndef BBMC_H
#define BBMC_H

#include <boost/dynamic_bitset.hpp>
#include <vector>
#include "Vertex.h"

class BBMC{
public:
	Vertex(int index, int degree);
	~Vertex();
	void init(int n);
	int applyColor();
	void removeColor(int i);

private:
	boost::dynamic_bitset<> N();
	boost::dynamic_bitset<> invN();
	std::vector<Vertex> V;
};

#endif