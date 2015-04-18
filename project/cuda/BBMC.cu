#include "BBMC.h"
#include <iostream>
#include "stdio.h"
#include <bitset>
#include <queue>
#include <boost/thread/thread.hpp>

__constant__ int numV;
__constant__ int numI;
__device__ long nodes;
__device__ int* globalMax;
__device__ unsigned int* globalC;
__device__ unsigned int* globalP;
__constant__ unsigned int* constN;
__constant__ unsigned int* constInvN;

// This is the declaration of the function that will execute on the GPU.
__device__ void printBitSet(unsigned int* bitset, int size);
__device__ int findFirstBit(unsigned int* bitset);
__device__ int getSetBitCount(unsigned int* bitset);
__device__ void intersectBitSet(unsigned int* bitset1, unsigned int* bitset2);
__device__ void flipBit(unsigned int& bitset, int pos);
__device__ void setBit(unsigned int& bitset, int pos);
__device__ void clearBit(unsigned int& bitset, int pos);
__device__ void copyBitSet(unsigned int* dest, unsigned int* src);

__global__ void maxCliqueP(int* currMax, unsigned int* N, unsigned int* invN, unsigned int* solution, unsigned int* max, unsigned int* devC,
 unsigned int* devP, unsigned int* devRecC, unsigned int* devRecP, unsigned int* devNewP, unsigned short* devU, unsigned short* devColor);
__device__ void recSearchP(unsigned int* solution, unsigned int* max, unsigned int* C, unsigned int* P,
 unsigned int* newP, unsigned int workArr[], unsigned short* U, unsigned short* color, int level);
__device__ void colorVertsP(unsigned int* P, unsigned short* U, unsigned short* color, unsigned int* work);
__device__ void copyBitSetP(unsigned int* dest, unsigned int* src);
__device__ void getSetBitCountP(unsigned int* bitset, unsigned int* work);
__device__ void findFirstBitP(unsigned int* bitset, unsigned int* work);
__device__ void intersectBitSetP(unsigned int* bitset1, unsigned int* bitset2);

// queue functions
__global__ void testQueue(int* queue);


void BBMC::luanchKernel(int threadId, unsigned int* hostN, unsigned int* hostInvN, unsigned int* retSol, unsigned int* retMax, int* currMax){
	std::cout << "new kernel launched: " << threadId << std::endl;

	unsigned int* devN, *devInvN, *devSolution, *devMax, *devRecP, *devRecC, *devNewP;
	unsigned short* devU, *devColor;
	thrust::device_vector<unsigned int> devC, devP;
	cudaError_t err;
	cudaEvent_t start, end;
	float elapsedTime;
	unsigned int* sol = new unsigned int[numInts * numBlocks];
	unsigned int* max = new unsigned int[numBlocks];

	cudaSetDevice(threadId);

	// get limit for stack and heap size
	err = cudaDeviceSetLimit(cudaLimitStackSize, 40048);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMalloc( (void**) &devN, numInts * n * sizeof(unsigned int));
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMalloc( (void**) &devInvN, numInts * n * sizeof(unsigned int));
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMalloc( (void**) &devSolution, numInts * sizeof(unsigned int) * numBlocks);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMalloc( (void**) &devMax, sizeof(unsigned int) * numBlocks);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMalloc( (void**) &devRecC, sizeof(unsigned int) * numInts * n * numBlocks);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMalloc( (void**) &devRecP, sizeof(unsigned int) * numInts * n * numBlocks);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMalloc( (void**) &devNewP, sizeof(unsigned int) * numInts * n * numBlocks);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	// allocate a Mb for each block for color and U arrays
	err = cudaMalloc( (void**) &devU, 548576 * numBlocks);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMalloc( (void**) &devColor, 548576 * numBlocks);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}

	// move the adjacency matrices to memory on the GPU
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
	err = cudaMemcpyToSymbol(constN, &devN, sizeof(unsigned int*), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMemcpyToSymbol(constInvN, &devInvN, sizeof(unsigned int*), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}

	// allocate inital nodes for each gpu
	devC = activeC[threadId];
	devP = activeP[threadId];
	std::cout << "Input size: " << devC.size() << std::endl;



	cudaEventCreate(&start);
	cudaEventCreate(&end);


	cudaEventRecord( start, 0 );
	maxCliqueP<<<numBlocks, numInts, memSize>>>(currMax, devN, devInvN, devSolution, devMax, thrust::raw_pointer_cast( &devC[0] ),
	 thrust::raw_pointer_cast( &devP[0] ), devRecC, devRecP, devNewP, devU, devColor);
    cudaEventRecord( end, 0 );
    cudaEventSynchronize( end );
    cudaEventElapsedTime( &elapsedTime, start, end );
    std::cout << threadId << " Kernel Time: " << elapsedTime << std::endl;

	// get solution back from kernel
  	err = cudaMemcpy(sol, devSolution, numInts * numBlocks * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "Error1: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
  	err = cudaMemcpy(max, devMax, numBlocks * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "Error2: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}

	// print out maxes found in cuda for each search
	int m = 0, index = 0;
	for(int i=0; i<numBlocks; i++){
		if(max[i] > m){
			m = max[i];
			index = i;
			// std::cout << "i: " << i << " max: " << max[i] << std::endl;
		}
	}

	// printIntArray(sol + index*(numInts), 1, numInts);

	// //unsigned int* cudaSol = new unsigned int[numInts];
	// std::cout << "CUDA SOLUTION: " << threadId << std::endl;
	// for(int currInt = 0; currInt < numInts; currInt++){
	// 	// loop over each bit in the int
	// 	for(int b=0; b<sizeof(unsigned int)*8; b++){
	// 		int shift = 1 << b;
	// 		int val = sol[currInt + index*(numInts)] & shift;
	// 		if(val != 0)
	// 			std::cout << V[32 * currInt + b].index + 1 << " ";
	// 	}
	// 	std::cout << " | ";
	// }
	// std::cout << std::endl;


	// place local solution into global solution
	retMax[threadId] = max[index];
	for(int i=0; i<numInts; i++){
		retSol[(threadId * numInts) + i] = sol[index*numInts + i];
	}
}


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

void BBMC::printBitSet(const boost::dynamic_bitset<>& bitset) const{
	for(int i=0; i<bitset.size(); i++){
		std::cout << bitset[i] << " ";
		if(i%32 == 0 && i != 0)
			std::cout << " | ";
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

__device__ int findFirstBit(unsigned int* bitset){
	int pos = 0, index = 0;;
	for(int i=0; i < numI; i++){
		pos = __ffs(bitset[i]);
		//printf("pos %d\n", pos);
		if(pos != 0){
			index = i;
			break;
		}
	}
	return pos + (index * 32) - 1;
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
	for(int i=0; i<numI; i++){
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

//////////////////////////// CUDA PARALLEL FUNCITONS /////////////////////////
void BBMC::searchParallel(){
	cudaError_t err;
	numInts = (n+sizeof(int)*8-1)/(sizeof(int)*8);
	memSize = (sizeof(unsigned int) * numInts);
	// get the number of devices
	err = cudaGetDeviceCount(&numDevices);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	std::cout << "Num Devices: " << numDevices << std::endl;
	numBlocks = n;
	activeC.resize(numDevices);
	activeP.resize(numDevices);

	// calculate the number of ints needed per vertex on gpu
	int r = n % (sizeof(int)*8);
	std::cout << numInts << " " << r << " " << n << " " << (sizeof(unsigned int)*8) << std::endl;
	unsigned int* hostN = new unsigned int[numInts*n];
	unsigned int* hostInvN = new unsigned int[numInts*n];
	unsigned int* sol = new unsigned int[numDevices * numInts];
	unsigned int* max = new unsigned int[numDevices];

	for(int i=0; i<n; i++){
		N[i].resize(n);
		invN[i].resize(n);
		V[i].index = i;
		V[i].degree = degree[i];

	}

	// order vertices
	orderVertices();

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

				if(r != 0 && i == numInts-1 && j == r)
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

	generateInitialNodes(numBlocks, numDevices);

	size_t stackLimit, heapLimit;
	err = cudaDeviceGetLimit(&stackLimit, cudaLimitStackSize);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaDeviceGetLimit(&heapLimit, cudaLimitMallocHeapSize);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	std::cout << "Stack Limit: " << stackLimit << " Heap Limit: " << heapLimit << std::endl;

	// allocate unified memory to share current max size
	int* currentMax;
	cudaMallocManaged(&currentMax, sizeof(int));
	*currentMax = 1;

	// create threads to launch a kernel on each gpu
	boost::thread threads[numDevices];

	// size for bitwise operations
	for(int i=0; i<numDevices; i++){
		// maxCliqueP<<<numBlocks, numInts, memSize>>>(devN[i], devInvN[i], devSolution[i], devMax[i], thrust::raw_pointer_cast( &devC[i][0] ),
		//  thrust::raw_pointer_cast( &devP[i][0] ), devRecC[i], devRecP[i], devNewP[i], devU[i], devColor[i]);
		threads[i] = boost::thread(&BBMC::luanchKernel, this, i, hostN, hostInvN, sol, max, currentMax);
	}

	// join threads
	for(int i=0; i<numDevices; i++){
		threads[i].join();
	} 

    // get a cuda error if there was one
    err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "Last Kernel Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}

	// print out maxes found in cuda for each search
	int m = 0, index = 0;
	for(int i=0; i<numDevices; i++){
		if(max[i] > m){
			m = max[i];
			index = i;
		}
	}
	std::cout << "MAX SIZE: " << *currentMax << std::endl;
	// printIntArray(sol + index*(numInts), 1, numInts);

	unsigned int* cudaSol = new unsigned int[numInts];
	std::cout << "CUDA SOLUTION FINAL: " << std::endl;
	for(int currInt = 0; currInt < numInts; currInt++){
		// loop over each bit in the int
		for(int b=0; b<sizeof(unsigned int)*8; b++){
			int shift = 1 << b;
			int val = sol[currInt + index*(numInts)] & shift;
			if(val != 0)
				std::cout << V[32 * currInt + b].index + 1 << " ";
		}
		std::cout << " | ";
	}
	std::cout << std::endl;
}

void BBMC::generateInitialNodes(int numBlocks, int numDevices){
	boost::dynamic_bitset<> C(n);
	boost::dynamic_bitset<> P(n);
	boost::dynamic_bitset<> newP;
	std::queue<boost::dynamic_bitset<>> activeCBitset;
	std::queue<boost::dynamic_bitset<>> activePBitset;
	int m, v;
	unsigned int c = 0, p = 0;
	int num = numBlocks * numDevices;

	// initialize starting node
	for(int i=0; i<C.size(); i++){
		C[i] = 0;
		P[i] = 1;
	}

	// generate initial branches for the graph (equal to number of vertices)
	m = P.count();
	int U[m];
	int color[m];
	BBColor(P, U, color);

	// iterate over the candidate set
	for(int i=m-1; i>= 0 && activeCBitset.size()<num-1; i--){

		// select a vertex from P and add it to the current clique
		// newP is set of vertices in the candidate set P that are adjacent to v
		newP = P;
		int v = U[i];
		C[v] = 1;

		// perform bitwise and (fast for set of element that reside in word boundaries)
		newP &= N[v];

		activeCBitset.push(C);
		activePBitset.push(newP);

		// remove v from P and C when returning
		C[v] = 0;
		P[v] = 0;
	}
	if(P.count() > 0){
		activeCBitset.push(C);
		activePBitset.push(P);
	}

	// expand nodes in queue to enqueue more sub branches and increase parallelism
	while(activeCBitset.size() < num){
		// get first nodes in queue
		C = activeCBitset.front();
		activeCBitset.pop();
		P = activePBitset.front();
		activePBitset.pop();

		// color the nodes for optimization
		m = P.count();
		BBColor(P, U, color);

		// iterate over part of the candidate set (lower i's usually get bounded out)
		for(int i=m-1; i>= m/2 && activeCBitset.size()<num-1; i--){

			// select a vertex from P and add it to the current clique
			// newP is set of vertices in the candidate set P that are adjacent to v
			newP = P;
			int v = U[i];
			C[v] = 1;

			// perform bitwise and (fast for set of element that reside in word boundaries)
			newP &= N[v];

			// add node to the queue for GPU
			activeCBitset.push(C);
			activePBitset.push(newP);

			// remove v from P and C when returning
			C[v] = 0;
			P[v] = 0;
		}
		// push the node that was initially popped off back on
		activeCBitset.push(C);
		activePBitset.push(P);
	}

	std::cout << std::endl << "num blocks: " << activeCBitset.size() << std::endl;

	// convert bitsets to GPU format

	int count = 0;
	int limit = numBlocks;
	int index = 0;
	while(activeCBitset.size() > 0){
		C = activeCBitset.front();
		newP = activePBitset.front();
		activeCBitset.pop();
		activePBitset.pop();
		for(int i=0; i<numInts; i++){
			c = 0;
			p = 0;
			for(int j=0; j<sizeof(unsigned int)*8; j++){
				//std::cout << "j " << j << " i " << i << std::endl;
				// assign each bit the the integer array
				c |= C[i*32 + j] << j;
				p |= newP[i*32 + j] << j;
				for(int b=0; b<32; b++){
					int num = (c & (1 << b));
					if(num != 0)
						num = 1;
					//std::cout << num << " ";
				}
				//std::cout << std::endl;
			}		
			// std::cout << "c " << c << " p " <<  p << std::endl;
			activeC[index].push_back(c);
			activeP[index].push_back(p);
		}
		// std::cout << count << " " << limit << " " << index << std::endl;
		count++;
		if(count == limit){
			index++;
			count = 0;
		}
	}


	// while(activeCBitset.size() > 0){
	// 	C = activeCBitset.front();
	// 	activeCBitset.pop();
	// 	P = activePBitset.front();
	// 	activePBitset.pop();
	// 	std::cout << "node C: " << std::endl;
	// 	printBitSet(C);
	// 	std::cout << "node P: " << std::endl;
	// 	printBitSet(P);
	// }
}

__global__ void maxCliqueP(int* currMax, unsigned int* devN, unsigned int* devInvN, unsigned int* solution, unsigned int* max, unsigned int* devC,
 unsigned int* devP, unsigned int* devRecC, unsigned int* devRecP, unsigned int* newP, unsigned short* devU,
 unsigned short* devColor) {

	//time1 = clock();
	__shared__ unsigned int* C;
	__shared__ unsigned int* P;
	extern __shared__ unsigned int workArr[];
	__shared__ unsigned short* U;
	__shared__ unsigned short* color;
	//numI = 1;
	//printf("numI: %u\n", numI);

	// have each block 
	if(threadIdx.x == 0){
		nodes = 0;
		max[blockIdx.x] = 0;
		C = devC + (numI * blockIdx.x);
		P = devP + (numI * blockIdx.x);
		U = devU + (blockIdx.x * 548576/sizeof(unsigned short));
		color = devColor + (blockIdx.x * 548576/sizeof(unsigned short));
		globalMax = currMax;
	}
	__syncthreads();

	copyBitSetP(devRecC + (blockIdx.x * numV * numI), C);
	copyBitSetP(devRecP + (blockIdx.x * numV * numI), P);
	copyBitSetP(newP + (blockIdx.x * numV * numI), P);

	__syncthreads();

	recSearchP(solution, max, devRecC + (blockIdx.x * numV * numI) + numI, 
		devRecP + (blockIdx.x * numV * numI) + numI, newP + (blockIdx.x * numV * numI), workArr, U, color, 0);
}

__device__ void recSearchP(
 unsigned int* solution, unsigned int* max, unsigned int* C, unsigned int* P, unsigned int* newP,
  unsigned int workArr[], unsigned short* U, unsigned short* color, int level){

	int newPNum, cNum;

	// copy C and P
	copyBitSetP(C, C-numI);
	copyBitSetP(P, newP);

 	__syncthreads();

	// ahve only a single thread incr the num nodes
	if(threadIdx.x == 0 && blockIdx.x == 0){
		nodes++;
	}

	newP += numI;

	// //printf("tid: %d, address: %p\n", threadIdx.x, &workArr[threadIdx.x]);
	// __syncthreads();

	getSetBitCountP(P, workArr);

	__syncthreads();

	int m = workArr[0];

	// if(threadIdx.x == 0)
	// 	printf("level: %d m: %d\n", level, m);

	colorVertsP(P, U, color, workArr);

	__syncthreads();

	// iterate over the candidate set
	for(int i=m-1; i>=0; i--){

		getSetBitCountP(C, workArr);

		__syncthreads();

		if(color[i] + workArr[0] <= *globalMax){
			return;
		}

		// copy the candidate set
		copyBitSetP(newP, P);

		// pick a candidate
		int v = U[i];
		if(threadIdx.x == 0){
			setBit(C[v/32], v%32);
		}

		// create the new candidate set
		intersectBitSetP(newP, constN + (v * numI));

		__syncthreads();

		// get the set bits for the candidate set and the current set
		getSetBitCountP(newP, workArr);

		__syncthreads();
		newPNum = workArr[0];
		

		getSetBitCountP(C, workArr);
		__syncthreads();

		cNum = workArr[0];
		

		// if maximal, check for maximum
		if(newPNum == 0 && cNum > *globalMax){

			if(threadIdx.x == 0){
				// save the new max size so that it is shared among blocks
				atomicMax(globalMax, cNum);
				max[blockIdx.x] = cNum;
			}
			copyBitSetP(solution + (blockIdx.x*numI), C);
		}
		else if(newPNum > 0){
			recSearchP(solution, max, C + numI, P + numI, newP, workArr, U + m, color + m, level + 1);
		}

		// remove v from P and C
		if(threadIdx.x == 0){
			clearBit(C[v/32], v%32);
			clearBit(P[v/32], v%32);
		}	
	}
}

__device__ void colorVertsP(unsigned int* P, unsigned short* U, unsigned short* color, unsigned int* work){

	// copy the candidate set
	//printf("thread: %d\n", threadIdx.x);
	__shared__ int colorClass;
	__shared__ int i;
	__shared__ unsigned int copyP[40];
	__shared__ unsigned int Q[40];
	copyBitSetP(copyP, P);

	// have main thread init values to zero
	if(threadIdx.x == 0){
		colorClass = 0;
		i = 0;
	}

	__syncthreads();

	// get num bits set (work is the start of shared memory to work on)
	getSetBitCountP(copyP, work);

	__syncthreads();

	while(work[0] != 0){

		if(threadIdx.x == 0){
			colorClass++;
		}

		__syncthreads();

		// copy the candidate set
		copyBitSetP(Q, copyP);

		__syncthreads();

		getSetBitCountP(Q, work);

		while(work[0] != 0){
			// return the index of the first set bit
			findFirstBitP(Q, work);

			// remove v from Q and copyP
			if(threadIdx.x == 0){
				clearBit(copyP[work[0]/32], work[0]%32);
				clearBit(Q[work[0]/32], work[0]%32);
			}

			__syncthreads();

			intersectBitSetP(Q, &constInvN[(work[0]) * numI]);

			__syncthreads();

			if(threadIdx.x == 0){
				U[i] = work[0];
				color[i++] = colorClass;
			}

			// recalculate set bits in Q for inner loop
			__syncthreads();
			getSetBitCountP(Q, work);
			__syncthreads();

		}

		// recalculate set bits in copyP for outer loop
		__syncthreads();
		getSetBitCountP(copyP, work);
		__syncthreads();
	}
}

__device__ void copyBitSetP(unsigned int* dest, unsigned int* src){
	dest[threadIdx.x] = src[threadIdx.x];
}

__device__ void getSetBitCountP(unsigned int* bitset, unsigned int* work){

	// printf("tid: %d, address: %p\n", threadIdx.x, &work[threadIdx.x]);
	// __syncthreads();
	// return;
	work[threadIdx.x] = __popc(bitset[threadIdx.x]);

	__syncthreads();

	int size = numI;
	for(int s=numI/2; s>0; s>>=1){
		__syncthreads();

	    if(threadIdx.x < s)
	      work[threadIdx.x] += work[threadIdx.x+s];

		// have the first thread do one additional add if the current size is odd
		if(size&0x0001 == 0x0001 && threadIdx.x == 0)
			work[threadIdx.x] += work[size-1];

		size /= 2;
	}
}

__device__ void findFirstBitP(unsigned int* bitset, unsigned int* work){

	// set min to highest value
	if(threadIdx.x == 0)
		work[0] = numI * 32;

	// have each thread get bit pos in int
	unsigned int pos = __ffs(bitset[threadIdx.x]);

	// set atomic min if bit was found
	if(pos != 0){
		// calculate overall position in bitstring and attempt to set to min
		pos += (threadIdx.x * 32) - 1;
		atomicMin(work, pos);
	}
}

__device__ void intersectBitSetP(unsigned int* bitset1, unsigned int* bitset2){
	bitset1[threadIdx.x] &= bitset2[threadIdx.x];
}

////////////////////////////// QUEUE FUNCTIONS /////////////////////////////
void BBMC::queueFcn(){
	int* devQueue;

	cudaError_t err = cudaMalloc( (void**) &devQueue, 1024 * sizeof(int));
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}

	testQueue<<<1, 1>>>(devQueue);
	cudaDeviceSynchronize();
}

__device__ int qMaxSize = 1024;
__device__ unsigned int pos;
__global__ void testQueue(int* queue){
	printf("queue\n");

	pos = 0;

	for(int i=0; i<100; i++){
		printf("enqueueing: %d\n", atomicAdd(&pos, 1));
	}

	for(int i=0; i<150; i++){
		printf("dequeueing: %d\n", atomicSub(&pos, 1));
	}
}

		// unsigned int c = 0, p = 0;
		// for(int i=0; i<numInts; i++){
		// 	for(int j=0; j<sizeof(unsigned int)*8; j++){
		// 		//std::cout << "j " << j << " i " << i << std::endl;
		// 		// assign each bit the the integer array
		// 		c |= C[i*32 + j] << j;
		// 		p |= newP[i*32 + j] << j;
		// 		for(int b=0; b<32; b++){
		// 			int num = (c & (1 << b));
		// 			if(num != 0)
		// 				num = 1;
		// 			//std::cout << num << " ";
		// 		}
		// 		//std::cout << std::endl;
		// 	}		
		// 	//std::cout << "c " << c << " p " <<  p << std::endl;
		// 	activeC.push_back(c);
		// 	activeP.push_back(p);
		// 	c = 0;
		// 	p = 0;
		// }