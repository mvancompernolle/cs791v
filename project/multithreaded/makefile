CC=clang++
CFLAGS=-c -std=c++11 -pthread -O3

all: project

project: main.o Vertex.o MC.o MC0.o MCQ.o MCSA.o MCSB.o BBMC.o
	$(CC) -pthread main.o Vertex.o MC.o MC0.o MCQ.o MCSA.o MCSB.o BBMC.o -o project

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

MC.o: MC.cpp
	$(CC) $(CFLAGS) MC.cpp

MC0.o: MC0.cpp
	$(CC) $(CFLAGS) MC0.cpp

MCQ.o: MCQ.cpp
	$(CC) $(CFLAGS) MCQ.cpp

MCSA.o: MCSA.cpp
	$(CC) $(CFLAGS) MCSA.cpp

MCSB.o: MCSB.cpp
	$(CC) $(CFLAGS) MCSB.cpp

BBMC.o: BBMC.cpp
	$(CC) $(CFLAGS) BBMC.cpp	

Vertex.o: Vertex.cpp
	$(CC) $(CFLAGS) Vertex.cpp

clean:
	rm *o project