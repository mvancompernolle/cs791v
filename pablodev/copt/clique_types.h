//clique_types.h: specific types for clique algorithms
//date: 13/02/15
//author: pss

#ifndef __CLIQUE_TYPES_H__
#define __CLIQUE_TYPES_H__


#include <iostream>
#include <string>

using namespace std;

//#define CLIQUE_MAX_INT	0x1FFFFFFF

#define NUM_ENUM_ALGORITHM 1
#define NUM_ENUM_INIT_COLOR 1
#define NUM_ENUM_INIT_ORDER 3

enum algorithm_t{DEF_ALG=0};
enum init_color_t{DEF_COL=0};

enum place_t{PLACE_FL=0,PLACE_LF};										//vertex placement
enum init_order_t {	NONE=0,	MIN_WIDTH, MIN_WIDTH_MIN_TIE_STATIC };		//sorting strategies

//parameter info
struct param_t{
	param_t():lb(0), ub(0),init_color(DEF_COL), init_order(MIN_WIDTH), alg(DEF_ALG), nThreads(1), time_limit(0), unrolled(false), filemane(""){};
	void print(ostream& o=cout){ 
		o<<"\n";
		o<<"lb: "<<lb<<endl;
		o<<"ub: "<<ub<<endl;
		o<<"icol: "<<init_color<<endl;
		o<<"iord: "<<init_order<<endl;
		o<<"alg: "<<alg<<endl;
		o<<"unrolled: "<<unrolled<<endl;
		o<<"nThreads: "<<nThreads<<endl;
		o<<"time_out: "<<time_limit<<endl;
		o<<"filename: "<<filemane<<endl;
	}
	
	int lb;										//initial solution (lower bound)
	int ub;										//initial upper bound (may be used for reduced initial allocation)
	init_color_t init_color;					//inital coloring strategy
	init_order_t init_order;				    //initial sorting strategy
	algorithm_t alg;						    //concrete variant
	bool unrolled;								//TRUE: initial loop unrolling
	int nThreads;
	int time_limit;							//seconds
	string filemane;											
};



#endif __CLIQUE_TYPES_H__





