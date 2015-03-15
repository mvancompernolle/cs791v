cpps=$(shell ls *.cpp)
headers=$(shell ls *.h)
sources=$(cpps) $(headers)
includes=-I. -I..

bbmc: $(sources)
	g++ $(cpps) $(includes) -DNDEBUG -march=native -std=c++0x -O3 -s -o bbmc -lrt
	
clean:
	rm bbmc

rebuild: clean bbmc
