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

#include "Dimacs.h"

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <algorithm>


Dimacs::Dimacs() : numVertices(0), adjacencyMatrixSizeLimit(1000000000 /* 1GB */) {
}

Dimacs::~Dimacs() {
}

bool Dimacs::load(const char* fname) {
    std::ifstream file(fname);
    std::string line;

    if (!file)
        return false;

    while (file) {
        std::getline(file, line);
        if (line.size() > 0) {
            std::istringstream ss(line);
            ss.ignore(2);
            if (line[0] == 'p') { // problem line
                std::string word;
                ss >> word;
                if (word != "edge")
                    return false;
                size_t nEdges = 0;
                ss >> numVertices >> nEdges;
                edges.reserve(nEdges);
                degrees.resize(numVertices, 0);
            } else if (line[0] == 'e') { // edge line
                int v1, v2;
                ss >> v1 >> v2;
                edges.push_back(std::make_pair(v1-1, v2-1));
                if (numVertices < (unsigned int)std::max(v1, v2)) {
                    numVertices = std::max(v1, v2);
                    degrees.resize(numVertices, 0);
                }
                degrees[v1-1]++;
                degrees[v2-1]++;
            }
        }
    }
    return true;
}


std::vector<std::vector<char> > Dimacs::getAdjacencyMatrix() const {
    if (adjacencyMatrixSizeLimit < getNumVertices() * getNumVertices())
        throw "Cannot create adjacency matrix because the number of vertices is to large";
    std::vector<std::vector<char> > matrix;
    matrix.resize(getNumVertices());
    for (auto& v : matrix) {
        v.resize(getNumVertices(), 0);
    }
        
    for (unsigned int ei = 0; ei < edges.size(); ++ei) {
        matrix[edges[ei].first][edges[ei].second] = 1;
        matrix[edges[ei].second][edges[ei].first] = 1;
    }
    
    return matrix;
}


void Dimacs::calculateGraphStats(int& maxDegree, int& minDegree, std::vector<float>& degreeHistogram) {
    auto degCopy = degrees;
    std::sort(degCopy.begin(), degCopy.end());
    size_t n = degCopy.size();
    size_t m = degreeHistogram.size();
    int cnt = 0;
    size_t di = n-1;
    for (size_t i = 0; i < m; ++i) {
        float bound = (m-i-1)*(float)n / m;
        while (degCopy[di] > bound && di > 0) {
            --di;
            ++cnt;
        }
        degreeHistogram[i] = cnt / (float)n;
        cnt = 0;
    }
    maxDegree = degCopy.front();
    minDegree = degCopy.back();
}

