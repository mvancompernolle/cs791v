CC=nvcc 
CFLAGS=-c -std=c++11 -arch=sm_35 -rdc=true 
LIBS= -lcudadevrt -lboost_thread -lboost_system

all: project

project: main.o MC.o MCQ.o BBMC.o Vertex.o
	$(CC) main.o MC.o MCQ.o BBMC.o Vertex.o -o project $(LIBS)

main.o: main.cu
	$(CC) $(CFLAGS) main.cu

MC.o: MC.cu
	$(CC) $(CFLAGS) MC.cu

MCQ.o: MCQ.cu
	$(CC) $(CFLAGS) MCQ.cu

BBMC.o: BBMC.cu
	$(CC) $(CFLAGS) BBMC.cu	

Vertex.o: Vertex.cu
	$(CC) $(CFLAGS) Vertex.cu

clean:
	rm *o project