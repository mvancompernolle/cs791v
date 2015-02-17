PA1 CS791v Read Me
===================================================

Parallel Mandelbrot
------------------------------
* By Matthew VanCompernolle

How to Run Sequential Code:
- go to sequential directory
- type: g++ -std=c++11 PA3_sequential.cpp
- type: ./a.out

Commands:
-s image size (NxN)
*to choose a image size to run type -s followed by the size which will act as both width and height*

How to Run Parallel Code:
- go to parallel directory
- type: make
- type: ./PA3

*ALL RUN OPTIONS CAN BE SELECTED FROM THE MENU SYSTEM*
- There are options to select the image size, number of threads, number of blocks, and max number of mandelbrot iterations
	- Each option prompts you for an initial size
	- You are then prompted if you want to loop on that option
	- if you choose yes, you will be prompted to enter an end size and your initial size will be the starting index of the loop (doubles each iteration)
	- if you choose no, your program will not loop on that paramater. It will just run all tests using that parameter size

