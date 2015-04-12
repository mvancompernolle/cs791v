#include "BBMC.h"
#include <iostream>
#include "stdio.h"
#include <bitset>

__constant__ int numV;
__constant__ int numI;
// Limit of about 700 vertices
__device__ __constant__ unsigned int constInvN[16000];
//__device__ float time1, time2;
__device__ long nodes;
__device__ int maxSize;
__device__ unsigned int* globalC;
__device__ unsigned int* globalP;
__device__ unsigned int* globalN;

// This is the declaration of the function that will execute on the GPU.
__device__ void printBitSet(unsigned int* bitset, int size);
__device__ int findFirstBit(unsigned int* bitset);
__device__ int getSetBitCount(unsigned int* bitset);
__device__ void intersectBitSet(unsigned int* bitset1, unsigned int* bitset2);
__device__ void flipBit(unsigned int& bitset, int pos);
__device__ void setBit(unsigned int& bitset, int pos);
__device__ void clearBit(unsigned int& bitset, int pos);
__device__ void copyBitSet(unsigned int* dest, unsigned int* src);

__global__ void maxCliqueP(unsigned int* N, unsigned int* solution,
 unsigned int* max, unsigned int* devC, unsigned int* devP, unsigned int* devRecC, unsigned int* devRecP, unsigned int* devNewP);
__device__ void recSearchP(unsigned int* N, unsigned int* solution, unsigned int* max, unsigned int* C, unsigned int* P, unsigned int* newP, unsigned int workArr[]);
__device__ void colorVertsP(unsigned int* P, unsigned int* U, unsigned int* color);
__device__ void copyBitSetP(unsigned int* dest, unsigned int* src);
__device__ void getSetBitCountP(unsigned int* bitset, unsigned int* work);
__device__ void findFirstBitP(unsigned int* bitset, unsigned int* work);
__device__ void intersectBitSetP(unsigned int* bitset1, unsigned int* bitset2);


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
	numInts = 0;
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

	for(int i=0; i<n; i++){
		N[i].resize(n);
		invN[i].resize(n);
		V[i].index = i;
		V[i].degree = degree[i];

	}

	// order vertices
	orderVertices();

/*
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
*/
	// calculate the number of ints needed per vertex on gpu
	numInts = (n+sizeof(int)*8-1)/(sizeof(int)*8);
	int r = n % (sizeof(int)*8);
	std::cout << numInts << " " << r << " " << n << " " << (sizeof(unsigned int)*8) << std::endl;
	unsigned int* hostN = new unsigned int[numInts*n];
	unsigned int* hostInvN = new unsigned int[numInts*n];
	unsigned int* sol = new unsigned int[numInts * n];
	unsigned int* max = new unsigned int[n];
	unsigned int* devN, *devInvN, *devSolution, *devMax;

	// need to preallocate memory for recursive calls
	unsigned int* devRecC, *devRecP, *devNewP;

	cudaSetDevice(1);

	// get limit for stack and heap size
	err = cudaDeviceSetLimit(cudaLimitStackSize, 50048);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2684354560);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
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

	err = cudaMalloc( (void**) &devN, numInts * n * sizeof(unsigned int));
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMalloc( (void**) &devSolution, numInts * sizeof(unsigned int) * n);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMalloc( (void**) &devMax, sizeof(unsigned int) * n);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMalloc( (void**) &devRecC, sizeof(unsigned int) * numInts * n * n);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMalloc( (void**) &devRecP, sizeof(unsigned int) * numInts * n * n);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMalloc( (void**) &devNewP, sizeof(unsigned int) * numInts * n * n);
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

	// move the adjacency matrices to constant memory on the GPU
	err = cudaMemcpy(devN, hostN, n * numInts * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
	err = cudaMemcpyToSymbol(constInvN, hostInvN, n * numInts * sizeof(unsigned int));
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

	generateInitialNodes();

	// allocated nodes on the gpu
	thrust::device_vector<unsigned int> devC = activeC;
	thrust::device_vector<unsigned int> devP = activeP;
	std::cout << "sizes: " << activeC.size() << " " << activeP.size() << std::endl;

	//printIntArray(hostN, n, numInts);
	//std::cout << std::endl;
	//printIntArray(hostInvN, n, numInts);
	cudaEvent_t start, end;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord( start, 0 );
	// size for bitwise operations + size of verts * 2
	int memSize = (sizeof(unsigned int) * numI) + (sizeof(unsigned int) * n * 2);
	maxCliqueP<<<n, numInts, memSize>>>(devN, devSolution, devMax, 
		thrust::raw_pointer_cast( &devC[0] ), thrust::raw_pointer_cast( &devP[0] ), devRecC, devRecP, devNewP);
    cudaEventRecord( end, 0 );
    cudaEventSynchronize( end );
    cudaEventElapsedTime( &elapsedTime, start, end );
    std::cout << "Kernel Time: " << elapsedTime << std::endl;

	// get solution back from kernel
  	err = cudaMemcpy(sol, devSolution, numInts * n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
  	err = cudaMemcpy(max, devMax, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}

	// print out maxes found in cuda for each search
	int m = 0, index = 0;
	for(int i=0; i<n; i++){
		if(max[i] > m){
			m = max[i];
			index = i;
			std::cout << "i: " << i << " max: " << max[i] << std::endl;
		}
	}
	printIntArray(sol + index*(numInts), 1, numInts);

	//unsigned int* cudaSol = new unsigned int[numInts];
	std::cout << "CUDA SOLUTION: " << std::endl;
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

void BBMC::generateInitialNodes(){
	boost::dynamic_bitset<> C(n);
	boost::dynamic_bitset<> P(n);
	boost::dynamic_bitset<> newP;

	for(int i=0; i<C.size(); i++){
		C[i] = 0;
		P[i] = 1;
	}

	int m = P.count();
	int U[m];
	int color[m];
	BBColor(P, U, color);

	// iterate over the candidate set
	for(int i=n-1; i>= 0; i--){

		// select a vertex from P and add it to the current clique
		// newP is set of vertices in the candidate set P that are adjacent to v
		newP = P;
		int v = U[i];
		C[v] = 1;

		// perform bitwise and (fast for set of element that reside in word boundaries)
		newP &= N[v];

		//boost::to_block_range(C, std::back_inserter(vecC));		
		//boost::to_block_range(newP, std::back_inserter(vecP));

				//hostN[(v*numInts) + i] |= (N[v][i * 32 + j] << j);

		//std::cout << "node C: " << std::endl;
		//printBitSet(C);
		//std::cout << "node P: " << std::endl;
		//printBitSet(newP);	

		unsigned int c = 0, p = 0;
		for(int i=0; i<numInts; i++){
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
			//std::cout << "c " << c << " p " <<  p << std::endl;
			activeC.push_back(c);
			activeP.push_back(p);
			c = 0;
			p = 0;
		}

		// remove v from P and C when returning
		C[v] = 0;
		P[v] = 0;
	}
}

__global__ void maxCliqueP(unsigned int* N, unsigned int* solution, unsigned int* max,
 unsigned int* devC, unsigned int* devP, unsigned int* devRecC, unsigned int* devRecP, unsigned int* newP) {

	//time1 = clock();
	__shared__ unsigned int* C;
	__shared__ unsigned int* P;
	extern __shared__ unsigned int workArr[];
	__shared__ unsigned int* U;
	//numI = 1;
	//printf("numI: %u\n", numI);

	// have each block 
	if(threadIdx.x == 0){
		//printf("const n: %u\n", constN[1]);
		nodes = 0;
		maxSize = 0;
		max[blockIdx.x] = 0;
		C = devC + (numI * blockIdx.x);
		P = devP + (numI * blockIdx.x);	
		U = workArr + numI;
	}

	copyBitSetP(devRecC + (blockIdx.x * numV * numI), C);
	copyBitSetP(devRecP + (blockIdx.x * numV * numI), P);
	copyBitSetP(newP + (blockIdx.x * numV * numI), P);
	recSearchP(N, solution, max, devRecC + (blockIdx.x * numV * numI) + numI, 
		devRecP + (blockIdx.x * numV * numI) + numI, newP + (blockIdx.x * numV * numI), workArr);


	//if(blockIdx.x == 0){
		//printf("Max found: %d\n", maxSize);
	//}
}

__device__ void recSearchP(unsigned int* N,
 unsigned int* solution, unsigned int* max, unsigned int* C, unsigned int* P, unsigned int* newP, unsigned int workArr[]){

	if(threadIdx.x == 0 && blockIdx.x == 0){
		printf("recursion called: \n");
	} 

	int newPNum, cNum;

 	__syncthreads();

	// copy C and P
	copyBitSetP(C, C-numI);
	copyBitSetP(P, newP);

 	__syncthreads();

	/*if(threadIdx.x == 0 && blockIdx.x == 0){
		printf("P: \n");
		printBitSet(P, numI);
		printf("\n");
	}*/

	// ahve only a single thread incr the num nodes
	if(threadIdx.x == 0 && blockIdx.x == 0){
		nodes++;
	}

	// use this to end testing
	if(nodes == 2)
		return;

	newP += numI;

	getSetBitCountP(P, workArr);
	int m = workArr[0];
	__shared__ unsigned int* U;
	__shared__ unsigned int* color;

	if(threadIdx.x == 0){
		// point U and color to their parts of shared memory		
		U = workArr + numI;
		color = U + numV;
	}

	__syncthreads();

	// color the vertices to create a tighter upper bound
	colorVertsP(P, U, color);

	__syncthreads();

	if(threadIdx.x == 0 && blockIdx.x == 0){
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
	}

	// iterate over the candidate set
	for(int i=m-1; i>=0; i--){

		getSetBitCountP(C, workArr);

		__syncthreads();

		if(color[i] + workArr[0] <= maxSize){
			return;
		}

		// copy the candidate set
		copyBitSetP(newP, P);

		// pick a candidate
		int v = U[i];
		if(threadIdx.x == 0){
			setBit(C[v/32], v%32);
		}

		__syncthreads();

		/*if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("newP: \n");
			printBitSet(newP, numI);
			printf("\n");
		}*/

		// create the new candidate set
		intersectBitSet(newP, &N[v * numI]);

		if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("new P: \n");
			printBitSet(newP, numI);
			printf("\n");
		}

		__syncthreads();

		// get the set bits for the candidate set and the current set
		getSetBitCountP(newP, workArr);
		newPNum = workArr[0];
		__syncthreads();

		getSetBitCountP(C, workArr);
		cNum = workArr[0];
		__syncthreads();

		// if maximal, check for maximum
		if(newPNum == 0 && cNum > maxSize){

			if(threadIdx.x == 0){
				// save the new max size so that it is shared among blocks
				atomicMax(&maxSize, cNum);
				//printf("b: %d, m: %d\n", blockIdx.x, getSetBitCount(C));
				max[blockIdx.x] = cNum;
				//printf("size: %d\n", maxSize);
			}

			copyBitSetP(solution + (blockIdx.x*numI), C);
			/*printf("Solution: ");
			printBitSet(solution + (blockIdx.x*numI), numI);
			printf("\n");*/
		}
		else if(newPNum > 0){
			//recSearchP(N, solution, max, C + numI, P + numI, newP, workArr);
		}

		// remove v from P and C
		if(threadIdx.x == 0){
			clearBit(C[v/32], v%32);
			clearBit(P[v/32], v%32);
		}	
	}
}

__device__ void colorVertsP(unsigned int* P, unsigned int* U, unsigned int* color){

	// copy the candidate set
	//printf("thread: %d\n", threadIdx.x);
	__shared__ int v;
	__shared__ int colorClass;
	__shared__ int i;
	__shared__ unsigned int copyP[32];
	__shared__ unsigned int Q[32];
	copyBitSetP(copyP, P);

	// have main thread init values to zero
	if(threadIdx.x == 0){
		v = 0;
		colorClass = 0;
	}

	__syncthreads();

	// get num bits set (U-numI is the start of shared memory to work on)
	getSetBitCountP(copyP, U-numI);

	__syncthreads();

	while(*(U-numI) != 0){

		if(threadIdx.x == 0){
			colorClass++;
		}

		__syncthreads();

		// copy the candidate set
		copyBitSetP(Q, copyP);
/*
if(threadIdx.x == 0 && blockIdx.x == 0){
	printf("Q: ");
	printBitSet(Q, numI);
	printf("\n");
}*/

		__syncthreads();

		getSetBitCountP(Q, U-numI);

		while(*(U-numI) != 0){
			// return the index of the first set bit
			findFirstBitP(Q, U-numI);

			/*if(threadIdx.x == 0 && blockIdx.x == 0){
				printf("bit pos: %d\n", *(U-numI));
			}*/
			//printf("%d - %d\n", colorClass, v);
			// remove v from Q and copyP
			if(threadIdx.x == 0){
				clearBit(copyP[*(U-numI)/32], *(U-numI)%32);
				clearBit(Q[*(U-numI)/32], *(U-numI)%32);
			}

			/*if(threadIdx.x == 0 && blockIdx.x == 0 && colorClass == 1){
				printf("Q: ");
				printBitSet(Q, numI);
				printf("\n");
			}*/

			__syncthreads();

			intersectBitSetP(Q, &constInvN[(*(U-numI)) * numI]);

			__syncthreads();

			/*if(threadIdx.x == 0 && blockIdx.x == 0 && colorClass == 1){
				printf("Q2: ");
				printBitSet(Q, numI);
				printf("\n");
			}*/

			if(threadIdx.x == 0){
				U[i] = *(U-numI);
				color[i++] = colorClass;
			}

			// recalculate set bits in Q for inner loop
			__syncthreads();
			getSetBitCountP(Q, U-numI);
			__syncthreads();

		}

		// recalculate set bits in copyP for outer loop
		__syncthreads();
		getSetBitCountP(copyP, U-numI);
		__syncthreads();
	}
}

__device__ void copyBitSetP(unsigned int* dest, unsigned int* src){
	dest[threadIdx.x] = src[threadIdx.x];
}

__device__ void getSetBitCountP(unsigned int* bitset, unsigned int* work){
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

	if(threadIdx.x == 0 && blockIdx.x == 0){
		//printf("count: %d\n", work[0]);
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

	/*if(C == NULL || P == NULL){
		printf("Out of heap memory\n");
		return;
	}
	if(oldP == NULL || oldC == NULL){
		printf("Out of stack memory\n");
		return;
	}	*/

	//printf("addr P: %p\n", oldP);

	/*if(nodes > 2)
		return;*/

	/*printf("cuda m: %d\n", m);
	printf("P: \n");
	printBitSet(P, numI);
	printf("\n");*/

	/*printf("cuda U:\n");
	for(int i=0; i<m; i++){
		printf("%u ", U[i]);
	}
	printf("\n");
	printf("cuda color:\n");
	for(int i=0; i<m; i++){
		printf("%u ", color[i]);
	}
	printf("\n");*/