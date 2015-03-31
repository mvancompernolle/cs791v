CC=nvcc
CFLAGS=-c -std=c++11

all: project

project: main.o MC.o cuda.o
	$(CC) main.o MC.o cuda.o -o project

main.o: main.cu
	$(CC) $(CFLAGS) main.cu

MC.o: MC.cu
	$(CC) $(CFLAGS) MC.cu

cuda.o: cuda.cu
	$(CC) $(CFLAGS) cuda.cu

clean:
	rm *o project