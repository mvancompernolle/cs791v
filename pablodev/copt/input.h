#include "clique_types.h"
#include "pablodev/tclap/CmdLine.h"

#include <iostream>

using namespace std;
using namespace TCLAP;


class input{
public:
	input(int argc_o, char** argv_o):argc(argc_o),argv(argv_o){
		
		for(int i=0; i<NUM_ENUM_INIT_COLOR; i++)
					allowed_init_color.push_back(i);
		for(int i=0; i<NUM_ENUM_ALGORITHM; i++)
					allowed_algorithms.push_back(i);
		for(int i=0; i<NUM_ENUM_INIT_ORDER; i++)
					allowed_init_order.push_back(i);
	
	}
	
	param_t parse(){
		param_t p;
		try{
			CmdLine cmd("Command description message", ' ', "0.9");
						
			ValueArg<int> lbArg("l","lower_bound","initial solution", false, 0,"int");
			cmd.add( lbArg);
			ValueArg<int> ubArg("u","upper_bound","initial upper bound", false, 0,"int");
			cmd.add( ubArg);

			ValuesConstraint<int> allowedColors(allowed_init_color);
			ValueArg<int> icolArg("c","init_col","initial coloring strategy", false, 0, &allowedColors);
			cmd.add( icolArg);

			ValuesConstraint<int> allowedSorting(allowed_init_order);
			ValueArg<int> iordArg("o","init_ord","initial sorting strategy", false, 1, &allowedSorting);		//default MIN_WIDTH
			cmd.add( iordArg);

			ValuesConstraint<int> allowedAlg(allowed_algorithms);
			ValueArg<int> algArg("a","algorithm","concrete algorithmic variant to run", false, 0, &allowedAlg);
			cmd.add( algArg);
					
			ValueArg<string> fileArg("f","filename","name of graph", false, "NO FILE","string");
			cmd.add( fileArg );

			SwitchArg unroll("r", "unrolling", "do loop unrolling", false);
			cmd.add( unroll);

			ValueArg<int> threadsArg("x","threads","number of threads", false, 1, "int");
			cmd.add( threadsArg);

			ValueArg<int> toutArg("t","timeout","time limit", false, 0, "int");
			cmd.add( toutArg);

			//parsing
			cmd.parse( argc, argv );


			//reading input
			p.lb=lbArg.getValue();
			p.ub=ubArg.getValue();
			p.init_color=(init_color_t)icolArg.getValue();
			p.init_order=(init_order_t)iordArg.getValue();
			p.alg=(algorithm_t)algArg.getValue();
			p.unrolled=unroll.getValue();
			p.nThreads=threadsArg.getValue();
			p.time_limit=toutArg.getValue();
			p.filemane=fileArg.getValue();
				
					
		}catch(ArgException &e) {
			cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
		}
		return p;
	}


private:
	int argc;
	char** argv;
	vector<int> allowed_init_order;
	vector<int> allowed_init_color;
	vector<int> allowed_algorithms;

	
};


