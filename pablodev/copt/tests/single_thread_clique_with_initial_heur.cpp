//single_thread_clique_with_initial_heur.cpp: binary to solve maximum clique in large sparse graphs with initial solution as parameter 
//PARAMS: <filename> <clique_size>
//author:pss
//date: 20/12/14

#include <iostream>
#include <stdlib.h>						 /* atoi */
#include "pablodev/utils/prec_timer.h"
#include "../clique.h"

using namespace std;

int main(int argc, char** argv){
	PrecisionTimer pt;
	double time_in_sec;
	
	if(argc!=3){
		cerr<<"incorrect number of arguments"<<endl;
		return -1;
	}

		
	//reading file
	string filename(argv[1]);
	cout<<"---------------------------------------------"<<endl;
	cout<<"READING:"<<filename<<endl;
	pt.wall_tic();
	sparse_ugraph usg(filename);
	time_in_sec=pt.wall_toc();
	usg.print_data();
	cout<<"[t:"<<time_in_sec<<"]"<<endl<<endl;
	
	//launch serial version
	Clique<sparse_ugraph> cusg(usg);
	cusg.set_initial_lb(atoi(argv[2]));					//sets initial clique size
	cout<<"SETUP"<<endl;
	pt.wall_tic();
	int sol=cusg.set_up_unrolled();
	time_in_sec=pt.wall_toc();
	cout<<"[t:"<<time_in_sec<<"]"<<endl<<endl;

	if(sol!=0){
		cout<<usg.get_name()<<" FOUND TRIVIAL SOLUTION DURING SETUP: "<<sol<<endl;
			return 0;
	}

	//search
	cout<<"SEARCH"<<endl;
	pt.wall_tic();
	cusg.run_unrolled();	
	time_in_sec=pt.wall_toc();

	cout<<"[t:"<<time_in_sec<<"]"<<"[w: "<<cusg.get_max_clique()<<"]"<<endl;
	cout<<"---------------------------------------------"<<endl;
}