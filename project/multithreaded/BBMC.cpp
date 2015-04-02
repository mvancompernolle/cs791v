#include "BBMC.h"
#include <iostream>
#include <thread>

/*
Algorithm description:
- Vertices are selected from the candidate set to add to the current clique in non-decreasing color order
	with a color cut-off.
- Bitset encoding of MCSa with the following features:
	1. The "BB" in "BB-MaxClique is for "Bit Board". Sets are represented using bit string.
	2. BBMC color sthe candidate set using a static sequential ordering, the ordering set at the top of search
	3. BBMC represents the neighborhood of a vertex and its inverse neighborhood as bit strings, rather 
		than using an adjacency matrix and its complement.
	4. When coloring takes place a color class perspective is taken, determining what vertices can be placed 
		in a color class together, before moving onto the next color class. 
*/

void call_from_thread(int tid) {
    std::cout << "Launched by thread " << tid << std::endl;
}

BBMC::BBMC(int n, std::vector<std::vector<int> > A, std::vector<int> degree, int style) : MCQ(n, A, degree, style){
	// N stores the neighborhood of vertex v
	// the set of vertices that are adjacent to v
	N = new boost::dynamic_bitset<>[n];
	// invN stores the inverse neighborhood of vertex v
	// the set of vertices that are not adjacent to v
	invN = new boost::dynamic_bitset<>[n];
	V.resize(n);
	numIdle = 0;
}

BBMC::~BBMC(){
	if(N != NULL)
		delete[] N;
	if(invN != NULL)
		delete[] invN;
}

void BBMC::search(){
	gettimeofday(&tod1, NULL);
	nodes = 0;

	// current clique encoded as a bit string
	boost::dynamic_bitset<> C(n);
	// candidate set encoded as a bit string
	boost::dynamic_bitset<> P(n);

	for(int i=0; i<n; i++){
		N[i].resize(n);
		invN[i].resize(n);
		V[i].index = i;
		V[i].degree = degree[i];

	}

	// order vertices
	orderVertices();
	for(int i=0; i<C.size(); i++){
		C[i] = 0;
		P[i] = 1;
	}
/*
std::cout << "Initial Ordering: " << std::endl;
std::cout << "C: " << std::endl;
printBitSet(C);
std::cout << "P: " << std::endl;
printBitSet(P);
std::cout << "N: " << std::endl;
for(int i=0; i<n; i++)
	printBitSet(N[i]);
std::cout << "invN: " << std::endl;
for(int i=0; i<n; i++)
	printBitSet(invN[i]);
std::cout << "V: (index / degree) " << std::endl;
for(Vertex& v: V){
	std::cout << v.index << "-" << v.degree << " ";
}
std::cout << std::endl;*/

    //Launch a group of threads
	int numThreads = 4;
    std::thread* t = new std::thread[numThreads];

    // launch a thread to populate the queue
	popThread = 0;
	BBMaxClique(C, P, 0);

    std::cout << "starting workers\n" << std::endl;

    // run all remaining work in the queue
    for(int i=0; i<numThreads; i++){
		t[i] = std::thread(&BBMC::runWorker, this, i);    	
    }
   

    //Join the threads with the main thread
    numIdle = numThreads;
    for (int i = 0; i < numThreads; ++i) {
       t[i].join();
    }

    delete[] t;

    //printQueue();

	//BBMaxClique(C, P);
/*
for(int& p: P)
	std::cout << p << " ";
std::cout << std::endl;
*/
}

void BBMC::runWorker(int tid){
	workItem w;

	// run worker threads while there is work left to do
	while(!work.empty()){
		//std::cout << "tid: " << tid << "size: " << work.size() << std::endl;
		idleMutex.lock();
		numIdle--;
		idleMutex.unlock();
		popMutex.lock();
		if(work.size() > 0){
			w = work.front();
			work.pop();
		}
		popMutex.unlock();
		BBMaxClique(w.C, w.P, tid);
		idleMutex.lock();
		numIdle++;
		idleMutex.unlock();
	}
}

void BBMC::orderVertices(){
	// calculate the sum of the neighboring degrees
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			if(A[i][j] == 1)
				V[i].setNebDeg(V[i].getNebDeg() + degree[j]);
		}
	}

	// order based on style passed in
	switch(style){
		case 1:
			// order by non-increasing degree, tie-breaking on index
			std::sort(V.begin(), V.end(), Vertex::VertexCmp);
			break;
		case 2:
			// order by minimum width order
			minWidthOrder(V);
			break;
		case 3:
			// known as MCR
			// order by non-increasing degree, tie-greaking on sum of the neighborhood
			// degree nubDeg and then on index
			std::sort(V.begin(), V.end(), Vertex::MCRComparator);
			break;
	}

	int u, v;
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			u = V[i].index;
			v = V[j].index;
			N[i][j] = (A[u][v] == 1);
			invN[i][j] = (A[u][v] == 0);
		}
	}

}

void BBMC::BBMaxClique(boost::dynamic_bitset<> C, boost::dynamic_bitset<> P, int tid){
	int w;

	// set populate to true if main thread and there is one element in C
	bool populate = false;
	if(tid == popThread && C.count() == 1)
		populate = true;
//std::cout << tid << populate << popThread << std::endl;

	// see if the time limit has been reached
	gettimeofday(&tod2, NULL);
	if(timeLimit > 0 && todiff(&tod2, &tod1)/1000 >= timeLimit) 
		return;

	// count the size of the backtrack search tree explored
	nodes++;

	int m = P.count();
	int U[m];
	int color[m];

	// builds color classes
	// if v = U[i] then color[i] is v's color and color[i] <= color[i+1]
	BBColor(P, U, color);

	// iterate over the candidate set
	for(int i=m-1; i>= 0; i--){

		// return if clique cannot grow large enough to be maximum clique
		if(color[i] + C.count() <= maxSize) 
			return;

		// select a vertex from P and add it to the current clique
		// newP is set of vertices in the candidate set P that are adjacent to v
		boost::dynamic_bitset<> newP(P);
		int v = U[i];
		C[v] = 1;

		// perform bitwise and (fast for set of element that reside in word boundaries)
		newP &= N[v];

		if(newP.none()){
			// if newP is empty is is maximal, so stop searching and save it if it is maximum
			if(C.count() > maxSize){
				saveSolution(C);
//std::cout << "here1" << std::endl;
			}
		}
		else{
//std::cout << "here2" << std::endl;
			// set populate to true if queue is empty and their are idle workers
			if(!populate && work.empty() && numIdle > 0){
				populate = true;
				popThread = tid;
			}
			if(populate){
//std::cout << "pop " << tid << std::endl;
				workItem w;
				w.C = C;
				w.P = newP;
				work.push(w);
			}
			else{
//std::cout << "expanding" << std::endl;
				BBMaxClique(C, newP, tid);
			}
		}
		// remove v from P and C when returning
		C[v] = 0;
		P[v] = 0;
	}
}

void BBMC::BBColor(const boost::dynamic_bitset<>& P, int U[], int color[]){
	// copy of candidate set
	boost::dynamic_bitset<> copyP(P);
	int v;
	int colorClass = 0;
	int i = 0;

	while(copyP.count() != 0){
		colorClass++;
		boost::dynamic_bitset<> Q(copyP);

		while(Q.count() != 0){
			// return the index of the first set bit
			v = Q.find_first();
			
			// remove v from Q and copyP
			copyP[v] = 0;
			Q[v] = 0;

			// perform a bitwise and operation
			// Q becomes set of vertices that are in Q but not adjacent to v
			Q &= invN[v];
			U[i] = v;
			color[i++] = colorClass;
		}
	}
}

void BBMC::saveSolution(const boost::dynamic_bitset<>& C){
	std::fill(solution.begin(), solution.end(), 0);
	for(int i=0; i<C.size(); i++){
		if(C[i])
			solution[V[i].index] = 1;
	}
	maxSize = C.count();
/*std::cout << "saved" << std::endl;
for(int i=0; i<solution.size(); i++){
	std::cout << solution[i] << " ";
}
std::cout << std::endl;*/
}

void BBMC::printBitSet(const boost::dynamic_bitset<>& bitset) const{
	for(int i=0; i<bitset.size(); i++){
		std::cout << bitset[i] << " ";
	}
	std::cout << std::endl;
}

void BBMC::printQueue(){
	while(!work.empty()){
		workItem w = work.front();
		std::cout << "QUEUE: " << std::endl << "C: ";
		printBitSet(w.C);
		std::cout << "P: ";
		printBitSet(w.P);
		work.pop();
	}
}