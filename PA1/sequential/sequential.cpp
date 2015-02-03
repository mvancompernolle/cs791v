#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
using namespace std;

#define NUM_ITERATIONS 5

// global variable for size of vector
int SIZE = 1000000;

long long int todiff(struct timeval *tod1, struct timeval *tod2)
{
	long long t1, t2;
	t1 = tod1->tv_sec * 1000000 + tod1->tv_usec;
	t2 = tod2->tv_sec * 1000000 + tod2->tv_usec;
	return t1 - t2;
}

void processFlags(int argc, char **argv){
	//cout << argc;
	if(argc > 2){

		for(int i=1; i<argc; i++){
			std::string arg = argv[i];
			if(arg == "-s")
				SIZE = atoi(argv[i+1]);

		}

	}
}


int main(int argc, char *argv[]){

	// process any flags from command line
	processFlags(argc, argv);

	int *vectorA, *vectorB, *vectorC;
	struct timeval tod1, tod2;
	double totalMsgTime = 0.0f, avgTime = 0.0f, minMsgTime = 0.0f, maxMsgTime = 0.0f, msgTime = 0.0f;
	ofstream fout;

	srand(time(NULL));

	vectorA = new int[SIZE];
	vectorB = new int[SIZE];
	vectorC = new int[SIZE];

	// fill vectors A and B with random info
	for(int i=0; i<SIZE; i++){
		vectorA[i] = rand() % 10;
		vectorB[i] = rand() % 10;
	}

	for(int i=0; i<NUM_ITERATIONS; i++){

		// get time as soon as message sends
		gettimeofday(&tod1, NULL);

		// add vectors A and B to C
		for(int index=0; index < SIZE; index++){
			vectorC[index] = vectorA[index] + vectorB[index];
		}

		// get time as soon as message is recieved
		gettimeofday(&tod2, NULL);

		// determine message passing time and add it to the average (in microseconds)
		msgTime = todiff(&tod2, &tod1);
		totalMsgTime += msgTime;
	}

	// delete vector data
	delete []vectorA;
	delete []vectorB;
	delete []vectorC;

	// calculate average message time
	avgTime = totalMsgTime/NUM_ITERATIONS;

	// print the average, minimum, and maximum message times
	std::cout << std::setprecision(4) << std::setiosflags(std::ios::fixed) << std::setiosflags(std::ios::showpoint) << "Avg Time Passed for size: "
		 << SIZE << " " << avgTime/1000 << " ms" << std::endl;

	// append results to the end of file
	fout.open("sequential_results.txt", ios::app);
	fout << SIZE << ", " << avgTime/1000 << endl;

	return 0;
}
