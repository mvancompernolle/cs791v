#ifndef VERTEX_H
#define VERTEX_H

class Vertex{
public:
	Vertex();
	~Vertex();
	//Vertex& operator= (const Vertex& source);
	void init(int n);
	int applyColor();
	void removeColor(int i);
	int getNebDeg();
	void setNebDeg(int n);
	static bool VertexCmp(const Vertex& a, const Vertex& b);
	static bool MCRComparator(const Vertex& a, const Vertex& b);
	int degree, index;
private:
	int color, saturation, nebDeg;
	bool* domain;
};

#endif