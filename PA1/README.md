PA1 CS791v Read Me
===================================================

Parallel Vector Addition
------------------------------
* By Matthew VanCompernolle

How to Run Sequential Code:
- go to sequential directory
- type: g++ -std=c++11 sequential.cpp
- type: ./a.out

Commands:
-s vectorSize
*to choose a vector size to run type -s followed by the size*

How to Run Parallel Code:
- go to parallel directory
- type: make
- type: ./PA1

*ALL USER INPUT IS DONE AS A COMMAND ARGUMENT*
*THE PROGRAM WILL RUN WITH DEFAULT IF NOTHING IS PASSED IN*

Commands:
-s vectorSize
*to choose a vector size to run type -s followed by the size*

-i methodType
*type -i followed by "normal" or "striding" to select the program method*

-io timeIOFlag
*type -io folowed by true or false to select whether to time the I/O in addition to run time*

-b blockSize
*type -b followed by a positive integer to choose a block size for the kernel*

-t threadSize
*type -t followed by a positive integer to choose a thread size for the kernel*

-ti minThreadSize maxThreadSize step
*type -ti followed by a minimum thread size, maximum thread size, and a step size to run the program in a loop

-bi minBlockSize maxBlockSize step
*type -bi followed by a minimum block size, maximum block size, and a step size to run the program in a loop