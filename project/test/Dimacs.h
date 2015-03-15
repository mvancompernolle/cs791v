/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 */

#ifndef DIMACS_H
#define DIMACS_H


#include <utility>
#include <vector>


class Dimacs {
public:
    typedef std::pair<int, int> Edge;
    
protected: public:
    std::vector<Edge> edges;
    std::vector<int> degrees;
    unsigned int numVertices;
    unsigned long long int adjacencyMatrixSizeLimit; // TODO: set this from commandline
    
public:
    Dimacs();
    ~Dimacs();
    
    bool load(const char* fname);
    unsigned int getNumVertices() const {return numVertices;}
    unsigned int getNumEdges() const {return edges.size();}
    std::vector<std::vector<char> > getAdjacencyMatrix() const;
    std::vector<int> getDegrees() const {return degrees;}
    void calculateGraphStats(int& maxdegree, int& minDegree, std::vector<float>& degreeHistogram);
};


#endif // DIMACS_H
