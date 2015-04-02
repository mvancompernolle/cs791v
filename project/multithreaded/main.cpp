#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>
#include <time.h>
#include "MC.h"
#include "MC0.h"
#include "MCQ.h"
#include "MCSA.h"
#include "MCSB.h"
#include "BBMC.h"

void readDIMACS(std::string fname,  std::vector<int>& degree, std::vector<std::vector<int> >& A, int& n);
long long int todiff(struct timeval *tod1, struct timeval *tod2);

int main(int argc, char** argv){

	std::cout << argc << " " << argv[1] << " " << argv[2] << std::endl;
	std::string implementation(argv[1]);

	std::vector<int> degree;
	std::vector<std::vector<int> > A;
	int n; 
	MC* mc = NULL;

	// read in the graph
	readDIMACS(argv[2], degree, A, n);

	// select the implemenation
	if(!implementation.compare("MC")){
		mc = new MC(n, A, degree);
	}
	else if(!implementation.compare("MC0")){
		mc = new MC0(n, A, degree);
	}
	else if(!implementation.compare("MCQ1")){
		mc = new MCQ(n, A, degree, 1);
	}
	else if(!implementation.compare("MCQ2")){
		mc = new MCQ(n, A, degree, 2);
	}
	else if(!implementation.compare("MCQ3")){
		mc = new MCQ(n, A, degree, 3);
	}
	else if(!implementation.compare("MCSA1")){
		mc = new MCSA(n, A, degree, 1);
	}
	else if(!implementation.compare("MCSA2")){
		mc = new MCSA(n, A, degree, 2);
	}
	else if(!implementation.compare("MCSA3")){
		mc = new MCSA(n, A, degree, 3);
	}
	else if(!implementation.compare("MCSB1")){
		mc = new MCSB(n, A, degree, 1);
	}
	else if(!implementation.compare("MCSB2")){
		mc = new MCSB(n, A, degree, 2);
	}
	else if(!implementation.compare("MCSB3")){
		mc = new MCSB(n, A, degree, 3);
	}
	else if(!implementation.compare("BBMC1")){
		mc = new BBMC(n, A, degree, 1);
	}
	else if(!implementation.compare("BBMC2")){
		mc = new BBMC(n, A, degree, 2);
	}
	else if(!implementation.compare("BBMC3")){
		mc = new BBMC(n, A, degree, 3);
	}
	else{
		std::cout << "Valid implementation was not selected" << std::endl;
		return 1;
	}

	// set time limit if passed in
	if(argc > 3)
		mc->setTimeLimit(1000 * atoi(argv[3]));

	timeval tod1, tod2;
	gettimeofday(&tod1, NULL);
	mc->search();
	gettimeofday(&tod2, NULL);

	// output results
	std::cout << " -------- Results --------- " << std::endl;
	std::cout << mc->getMaxSize() << " " << mc->getNumNodes() << " " << todiff(&tod2, &tod1)/1000 << std::endl;
	std::vector<int> sol = mc->getSolution();
	for(int i=0; i<sol.size(); i++){
		if(sol[i] == 1)
			std::cout << i+1 << " ";
	}
	std::cout << std::endl;

	// output results to a file
	std::ofstream fout;
	fout.open("results.txt", std::ofstream::app);
	std::string timeLimit;
	if(argv[3] != NULL){
		timeLimit = argv[3];
	}
	else{
		timeLimit = "-1";
	}
	//fout << argv[1] << ", " << argv[2] << ", " << timeLimit << std::endl;
	//fout << mc->getMaxSize() << ", " << mc->getNumNodes() << ", " << todiff(&tod2, &tod1)/1000 << std::endl << std::endl;	

	// delete max clique algorithm
	delete mc;

	return 0;
}

void readDIMACS(std::string fname, std::vector<int>& degree, std::vector<std::vector<int> >& A, int& n){
	std::string s = "";
	int m, i, j;
	int counter = 0;
	std::ifstream fin;

	fin.open(fname);

	if(fin.good()){
		while(fin.good() && s.compare("p")){
			fin >> s;
		}

		fin >> s;
		fin >> n;
		fin >> m;
		degree.resize(n);
		A.resize(n);
		for(int i=0; i<n; i++)
			A[i].resize(n);

		while(fin.good()){
			fin >> s;
			fin >> i;
			fin >> j;
			i--;
			j--;
			degree[i]++;
			degree[j]++;
			A[i][j] = A[j][i] = 1;
			counter++;
		}

		fin.close();
	}
	else{
		std::cout << "file not found" << std::endl;
		exit(1);
	}
}

long long int todiff(struct timeval *tod1, struct timeval *tod2)
{
	long long t1, t2;
	t1 = tod1->tv_sec * 1000000 + tod1->tv_usec;
	t2 = tod2->tv_sec * 1000000 + tod2->tv_usec;
	return t1 - t2;
}