# a simple makefile for building the sample program.
# I use multiple versions of gcc, but cuda only supports
# gcc 4.4 or lower. The ccbin command tells nvcc to use
# gcc 4.4 as its regular (non-gpu) compiler.
#
# the uncommented line should do the trick for you.

all: PA3

PA3: PA3.cu mandelbrot.cu
#	nvcc -ccbin=/home/richard/bin/ $^ -o $@
	nvcc $^ -o $@

clean:
	rm -f *.o
	rm -f *~
	rm PA3
