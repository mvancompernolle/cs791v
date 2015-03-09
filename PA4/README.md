PA1 CS791v Read Me
===================================================

Parallel Vector Reduction Part 2
------------------------------
* By Matthew VanCompernolle

How to Run Sequential Code:
- go to sequential directory
- type: g++ -std=c++11 PA4_sequential.cpp
- type: ./a.out

Commands:
-s vectorSize
*to choose a vector size to run type -s followed by the size*

How to Run Parallel Code:
- go to parallel directory
- type: make
- type: ./PA4

*ALL RUN OPTIONS CAN BE SELECTED FROM THE MENU SYSTEM*
- There are options to select the vector size, number of threads, number of blocks
	- Each opiton prompt you for an initial size
	- You are then prompted if you want to loop on that option
	- if you choose yes, you will be prompted to enter an end size and your initial size will be the starting index of the loop
	- if you choose no, your program will not loop on that paramater. It will just run all tests using that parameter size
- There is an option to select the recution type
	- CPU reduces the partial sums of the blocks on the CPU
	- Recursive Host recalls the kernel from the CPU until the final result is computed
	- Recursive Device recursively calls the kernel from the GPU until the final result is computed

