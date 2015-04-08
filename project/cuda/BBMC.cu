#include "BBMC.h"
#include <iostream>
#include "stdio.h"

__constant__ int numV;
__constant__ int numI;
//__device__ float time1, time2;
__device__ long nodes;
__device__ int maxSize;


// This is the declaration of the function that will execute on the GPU.
__global__ void maxClique(unsigned int* N, unsigned int* invN, unsigned int* solution);
__device__ void recSearch(unsigned int* C, unsigned int* P, unsigned int* N, unsigned int* invN, unsigned int* solution);
__device__ void colorVerts(unsigned int* P, unsigned int* U, unsigned int* color, unsigned int* invN);
__device__ void printBitSet(unsigned int* bitset, int size);
__device__ int findFirstBit(unsigned int* bitset);
__device__ int getSetBitCount(unsigned int* bitset);
__device__ void intersectBitSet(unsigned int* bitset1, unsigned int* bitset2);
__device__ void flipBit(unsigned int& bitset, int pos);
__device__ void setBit(unsigned int& bitset, int pos);
__device__ void clearBit(unsigned int& bitset, int pos);
__device__ void copyBitSet(unsigned int* dest, unsigned int* src);

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

BBMC::BBMC(int n, std::vector<std::vector<int> > A, std::vector<int> degree, int style) : MCQ(n, A, degree, style){
	// N stores the neighborhood of vertex v
	// the set of vertices that are adjacent to v
	N = new boost::dynamic_bitset<>[n];
	// invN stores the inverse neighborhood of vertex v
	// the set of vertices that are not adjacent to v
	invN = new boost::dynamic_bitset<>[n];
	V.resize(n);
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


std::cout << "N: " << std::endl;
for(int i=0; i<n; i++)
	printBitSet(N[i]);
std::cout << "invN: " << std::endl;
for(int i=0; i<n; i++)
	printBitSet(invN[i]);
std::cout << std::endl;
std::cout << "V: (index / degree) " << std::endl;
for(Vertex& v: V){
	std::cout << v.index << "-" << v.degree << " ";
}
std::cout << std::endl;

	// calculate the number of ints needed per vertex on gpu
	int numInts = (n+sizeof(int)*8-1)/(sizeof(int)*8);
	int r = n % (sizeof(int)*8);
	std::cout << numInts << " " << r << " " << n << " " << (sizeof(unsigned int)*8) << std::endl;
	unsigned int* hostN = new unsigned int[numInts*n];
	unsigned int* hostInvN = new unsigned int[numInts*n];
	unsigned int* sol = new unsigned int[numInts];
	unsigned int* devN, *devInvN, *devSolution;


	cudaSetDevice(1);
	cudaError_t err = cudaMalloc( (void**) &devN, numInts * n * sizeof(unsigned int));
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMalloc( (void**) &devInvN, numInts * n * sizeof(unsigned int));
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMalloc( (void**) &devSolution, numInts * sizeof(unsigned int));
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}

	// default all values to 0
	for(int r=0; r<n; r++){
		for(int i=0; i<numInts; i++){
			hostN[r*numInts + i] = 0;
			hostInvN[r*numInts + i] = 0;
		}
	}

	// copy the adjacency matrices into 1D arrays
	for(int v=0; v<n; v++){
		for(int i=0; i<numInts; i++){
			for(int j=0; j<sizeof(unsigned int)*8; j++){

				if(i == numInts-1 && j == r)
					break;

				// assign each bit the the integer array
				hostN[(v*numInts) + i] |= (N[v][i * 32 + j] << j);

				int invVal = 0;
				if(N[v][i * 32 + j] == 0)
					invVal = 1;
				hostInvN[(v*numInts) + i] |= (invVal << j);
			}
		}
	}

	/*// find the first set bit of the adjacency matrix
	int res;
	static const int MultiplyDeBruijnBitPosition[32] = 
	{
	  0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
	  31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
	};
	res = MultiplyDeBruijnBitPosition[((uint32_t)((hostN[0] & -hostN[0]) * 0x077CB531U)) >> 27];
	std::cout << "first bit " << r << std::endl; */

	// move the adjacency matrices to constant memory on the GPU
	err = cudaMemcpy(devN, hostN, n * numInts * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMemcpy(devInvN, hostInvN, n * numInts * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}

	// move number of vertices to constant memory
	err = cudaMemcpyToSymbol(numV, &n, sizeof(int), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMemcpyToSymbol(numI, &numInts, sizeof(int), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}

	//printIntArray(hostN, n, numInts);
	//std::cout << std::endl;
	//printIntArray(hostInvN, n, numInts);
	maxClique<<<1, 1>>>(devN, devInvN, devSolution);
	cudaDeviceSynchronize();

	// get solution back from kernel
  	err = cudaMemcpy(sol, devSolution, numInts * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}

	//unsigned int* cudaSol = new unsigned int[numInts];
	std::cout << "CUDA SOLUTION: " << std::endl;
	for(int currInt = 0; currInt < numInts; currInt++){
		// loop over each bit in the int
		for(int b=0; b<sizeof(unsigned int)*8; b++){
			int shift = 1 << b;
			int val = sol[currInt] & shift;
			if(val != 0)
				std::cout << V[32 * currInt + b].index + 1 << " ";
		}
		std::cout << " | ";
	}
	std::cout << std::endl;

	BBMaxClique(C, P);
/*
for(int& p: P)
	std::cout << p << " ";
std::cout << std::endl;
*/
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

void BBMC::BBMaxClique(boost::dynamic_bitset<> C, boost::dynamic_bitset<> P){
	int w;
/*
std::cout << std::endl;
printBitSet(C);
printBitSet(P);
*/
	// see if the time limit has been reached
	gettimeofday(&tod2, NULL);
	if(timeLimit > 0 && todiff(&tod2, &tod1)/1000 >= timeLimit) 
		return;

	// count the size of the backtrack search tree explored
	nodes++;

	int m = P.count();
	int U[m];
	int color[m];

//std::cout << m << std::endl;

	// builds color classes
	// if v = U[i] then color[i] is v's color and color[i] <= color[i+1]
	BBColor(P, U, color);


std::cout << "host" << std::endl << "m: " << m << std::endl;
/*std::cout << "C: " << std::endl;
printBitSet(C);
std::cout << "P: " << std::endl;
printBitSet(P);*/
std::cout << "U: " << std::endl;
printArray(U, m);
std::cout << "color: " << std::endl;
printArray(color, m);
std::cout << std::endl;


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
/*
std::cout << "N: ";
printBitSet(N[v]);
std::cout << "newP: ";
printBitSet(newP);
*/
		// if newP is empty is is maximal, so stop searching and save it if it is maximum
		if(newP.none() && C.count() > maxSize){
			saveSolution(C);
		}
		// else recursively continue search 
		else if(!newP.none()){
//std::cout << "called again" << std::endl;
			BBMaxClique(C, newP);
		}

		// remove v from P and C when returning
		C[v] = 0;
		P[v] = 0;

		//gettimeofday(&t2, NULL);
		//std::cout << todiff(&t2, &t1)/1000 << std::endl;
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
	std::cout << "Sol C: " << std::endl;
	printBitSet(C);
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

void BBMC::printIntArray(unsigned int* arr, int n, int numInts) const{
	// loop over each row
	for(int i=0; i<n; i++){
		// loop over each int
		for(int currInt = 0; currInt < numInts; currInt++){
			// loop over each bit in the int
			for(int b=0; b<sizeof(unsigned int)*8; b++){
				int shift = 1 << b;
				int val = arr[(i*numInts) + currInt] & shift;
				if(val != 0)
					val = 1;
				std::cout << val << " ";
			}
			std::cout << " | ";
		}
		std::cout << std::endl;
	}
}

////////////////////// CUDA FUNCTIONS ///////////////////////////////////////////////////////////
__global__ void maxClique(unsigned int* N, unsigned int* invN, unsigned int* solution) {

	//time1 = clock();
	nodes = 0;
	maxSize = 0;
	//numI = 1;
	//printf("numI: %u\n", numI);

	unsigned int* C = new unsigned int[numI];
	unsigned int* P = new unsigned int[numI];
	for(int i=0; i<numI; i++){
		C[i] = 0;
		if(i != numI -1 || numV%32 == 0)
			P[i] = ~0;
		else{
			P[i] = (1 << (numV%32)) - 1;
		}
	}
	printBitSet(C, numI);
	printf("\nnumI: %d\n", numI);
	printBitSet(P, numI);
	printf("\n");

	recSearch(C, P, N, invN, solution);

	delete[] C;
	delete[] P;
}

__device__ void recSearch(unsigned int* oldC, unsigned int* oldP, unsigned int* N, unsigned int* invN, unsigned int* solution){

	// copy C and P
	unsigned int* C = new unsigned int[numI];
	copyBitSet(C, oldC);
	unsigned int* P = new unsigned int[numI];
	copyBitSet(P, oldP);

	nodes++;

	int m = getSetBitCount(P);
	unsigned int* U = new unsigned int[m];
	unsigned int* color = new unsigned int[m];
	printf("cuda m: %d\n", m);

	colorVerts(P, U, color, invN);

	printf("cuda U:\n");
	for(int i=0; i<m; i++){
		printf("%u ", U[i]);
	}
	printf("\n");
	printf("cuda color:\n");
	for(int i=0; i<m; i++){
		printf("%u ", color[i]);
	}
	printf("\n");

	// iterate over the candidate set
	for(int i=m-1; i>=0; i--){

		if(color[i] + getSetBitCount(C) <= maxSize){
			return;
		}

		// copy the candidate set
		unsigned int* newP = new unsigned int[numI];
		copyBitSet(newP, P);
		int v = U[i];
		setBit(C[v/32], v%32);

		intersectBitSet(newP, &N[v * numI]);

		// if maximal, check for maximum
		if(getSetBitCount(newP) == 0 && getSetBitCount(C) > maxSize){
			// save the solution
			maxSize = getSetBitCount(C);
			printf("size: %d\n", maxSize);
			copyBitSet(solution, C);
			printf("Solution: ");
			printBitSet(solution, numI);
			printf("\n");
		}
		else if(getSetBitCount(newP) > 0){
			recSearch(C, newP, N, invN, solution);
		}

		// remove v from P and C
		clearBit(C[v/32], v%32);
		clearBit(P[v/32], v%32);

	}

	delete[] C;
	delete[] P;
}

__device__ void colorVerts(unsigned int* P, unsigned int* U, unsigned int* color, unsigned int* invN){
	// copy the candidate set
	unsigned int* copyP = new unsigned int[numI];
	copyBitSet(copyP, P);
	int v;
	int colorClass = 0;
	int i = 0;

	while(getSetBitCount(copyP) != 0){
		colorClass++;

		// copy the candidate set
		unsigned int* Q = new unsigned int[numI];
		copyBitSet(Q, copyP);

		while(getSetBitCount(Q) != 0){
			// return the index of the first set bit
			v = findFirstBit(Q);
			//printf("%d - %d\n", colorClass, v);
			// remove v from Q and copyP
			clearBit(copyP[v/32], v%32);
			clearBit(Q[v/32], v%32);

			intersectBitSet(Q, &invN[v * numI]);

			U[i] = v;
			color[i++] = colorClass;
		}
	}
}

__device__ int findFirstBit(unsigned int* bitset){
	int pos = 0;
	for(int i=0; i < numI; i++){
		pos = (i*32) + __ffs(bitset[i]);
		if(pos != 0)
			break;
	}
	return pos - 1;
}

__device__ int getSetBitCount(unsigned int* bitset){
	int count = 0;
	for(int i=0; i<numI; i++){
		count += __popc(bitset[i]);
	}
	return count;
}

__device__ void intersectBitSet(unsigned int* bitset1, unsigned int* bitset2){
	for(int i=0; i<numI; i++){
		bitset1[i] &= bitset2[i];
	}
}

__device__ void flipBit(unsigned int& bitset, int pos){
	bitset ^= (1u << pos);
}

__device__ void setBit(unsigned int& bitset, int pos){
	bitset |= (1u << pos);
}

__device__ void clearBit(unsigned int& bitset, int pos){
	bitset &= ~(1u << pos);
}

__device__ void copyBitSet(unsigned int* dest, unsigned int* src){
	for(int i=0; i<numV * numI; i++){
		dest[i] = src[i];
	}
}

__device__ void printBitSet(unsigned int* bitset, int size){

	// loop over each int
	for(int currInt = 0; currInt < size; currInt++){
		// loop over each bit in the int
		for(int b=0; b<sizeof(unsigned int)*8; b++){
			int shift = 1 << b;
			int val = bitset[currInt] & shift;
			if(val != 0)
				val = 1;
			printf("%u ", val);
		}
		printf(" | ");
	}
}


	/*unsigned int* tmp = N;
	if(threadIdx.x == 0 && blockIdx.x == 0){
		for(int i=0; i<numV; i++){
		  printBitSet(tmp, numI);
		  printf("\n");
		  tmp+=numI;
		}
	}

	// copy n to new bitset
	unsigned int* bitset = new unsigned int[numI * numV];
	copyBitSet(bitset, N);
	printf("\n\nBitset:\n");
	tmp = bitset;
	for(int i=0; i<numV; i++){
	  printBitSet(tmp, numI);
	  printf("\n");
	  tmp+=numI;
	}*/