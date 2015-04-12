#include <iostream>
#include "Vertex.h"
using namespace std;

Vertex::Vertex(){
	// index in P
	this->index = 0;
	// degree of the vertex index
	this->degree = 0;
	// sum of the degrees of the vertices in the neighborhood of vertex index
	nebDeg = 0;
	domain = NULL;
}

Vertex::~Vertex(){
	if(domain != NULL)
		delete[] domain;
	domain = NULL;
}

/*Vertex& Vertex::operator= (const Vertex& source){
	if(this == &source)
		return *this;

	index = source.index;
	degree = source.degree;
	color = source.color;
	saturation = source.saturation;
	nebDeg = source.nebDeg;

	if(domain != NULL)
		delete[] domain;
	domain = NULL;
}*/

void Vertex::init(int n){
	domain = new bool[n];
	std::fill(domain, domain+n, true);
	color = -1;
	saturation = 0;
}

int Vertex::applyColor(){
	for(color=0; !domain[color]; color++);
	domain[color] = false;
	return color;
}

void Vertex::removeColor(int i){
	if(domain[i]){
		saturation++;
		domain[i] = false;
	}
}

int Vertex::getNebDeg(){
	return nebDeg;
}

void Vertex::setNebDeg(int n){
	nebDeg = n;
}

bool Vertex::VertexCmp(const Vertex& a, const Vertex& b){
	return !(a.degree < b.degree || (a.degree == b.degree && a.index > b.index));
}

bool Vertex::MCRComparator(const Vertex& a, const Vertex& b){
	return !(a.degree < b.degree || (a.degree == b.degree && a.nebDeg < b.nebDeg)
		|| (a.degree == b.degree && a.nebDeg == b.nebDeg && a.index > b.index));
}