//single_thread_bbmc.cpp: bbmc algorithm [1][2] for exact maximum clique 
//[1] 
//[2]
//PARAMS: <filename> 
//author:pss
//date: 01/07/14

#include <iostream>
#include <stdlib.h>							 /* atoi */
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
	ugraph ug(filename);
	time_in_sec=pt.wall_toc();
	ug.print_data();
	cout<<"[t:"<<time_in_sec<<"]"<<endl<<endl;
	
	//launch serial version
	Clique<ugraph> cug(ug);
	cout<<"SETUP"<<endl;
	pt.wall_tic();
	cug.set_initial_lb(atoi(argv[2]));					//sets initial clique size
	int sol=cug.set_up();
	time_in_sec=pt.wall_toc();

	cout<<"[t:"<<time_in_sec<<"]"<<endl<<endl;

	if(sol!=0){
		cout<<ug.get_name()<<" FOUND TRIVIAL SOLUTION DURING SETUP: "<<sol<<endl;
			return 0;
	}

	//search
	cout<<"SEARCH"<<endl;
	pt.wall_tic();
	cug.run();	
	time_in_sec=pt.wall_toc();

	cout<<"[t:"<<time_in_sec<<"]"<<"[w: "<<cug.get_max_clique()<<"]"<<endl;
	cout<<"---------------------------------------------"<<endl;
}