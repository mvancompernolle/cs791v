#include "MCSB.h"
#include <iostream>
#include <algorithm>
#include <stack>

MCSB::MCSB(int n, std::vector<std::vector<int> > A, std::vector<int> degree, int style): MCSA(n, A, degree, style){
}

MCSB::~MCSB(){

}

void MCSB::numberSort(std::vector<int> C, std::vector<int> colOrd, std::vector<int>& P, int color[]){
	// records the number of colors used
	int delta = maxSize - C.size();
	int colors = 0;
	int m = colOrd.size();

	// clear out the color classes that might be used
	for(int i=0; i<m; i++){
		colorClass[i].clear();
	}

	// vertices are selected from colOrd and placed into first color class in which there are no conflicts
	for(int i=0; i<m; i++){
		int v = colOrd[i];
		int k = 0;
		// vertice in colorClass are not pair wise adjacent and have same color (independent set)
		while (conflicts(v, colorClass[k]))
			k++;
		colorClass[k].push_back(v);
		colors = std::max(colors, k+1);
		if(k+1 > delta && colorClass[k].size() == 1 && repair(v,k))
			colors--;
	}

/*
std::cout << "ColorClasses: " << std::endl;
for(int i=0; i<colors; i++){
	std::cout << "C" << i << " ";
	for(int j=0; j<colorClass[i].size(); j++){
		std::cout << colorClass[i][j] << " ";
	}
	std::cout << std::endl;
}
*/
	// pidgeon hole sort
	P.clear();
	int i = 0;
	for(int k=0; k<colors; k++){
		for(int j=0; j<colorClass[k].size(); j++){
			int v = colorClass[k][j];
			P.push_back(v);
			color[i++] = k+1;
		}
	}

/*
std::cout << "Pidgeon Hole: ";
for(int& v: P) std::cout << v << " ";
std::cout << std::endl;

std::cout << "Colors: ";
for(int i=0; i<m; i++) std::cout << color[i] << " ";
std::cout << std::endl;
*/
}

bool MCSB::repair(int v, int k){
	int w;
	for(int i=0; i<k-1; i++){
		w = getSingleConflictVariable(v, colorClass[i]);
		if(w >= 0){
			for(int j=i+1; j<k; j++){
				if(!conflicts(w, colorClass[j])){
					colorClass[k].erase(std::remove(colorClass[k].begin(), colorClass[k].end(), v), colorClass[k].end());
					colorClass[i].erase(std::remove(colorClass[i].begin(), colorClass[i].end(), w), colorClass[i].end());
					colorClass[i].push_back(v);
					colorClass[j].push_back(w);
					return true;
				}
			}
		}
	}
	return false;
}

int MCSB::getSingleConflictVariable(int v, std::vector<int> cClass){
	int conflictVar = -1;
	int count = 0;
	for(int i=0; i<cClass.size() && count < 2; i++){
		int w = cClass[i];
		if(A[v][w] == 1){
			conflictVar = w;
			count++;
		}
	}
	if(count > 1)
		return -count;
	return conflictVar;
}