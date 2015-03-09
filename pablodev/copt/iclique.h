//////////////////////////////////
// itests.h , interface class for testing algorithms
#pragma once

class Result;
class GraphBB;
struct param_t;
struct setup_t;
struct param_col_t;

class IClique{
public:
	void set_param(param_t p){param=p;}
	param_t& get_param(){return param;}

	virtual Result run()=0;	
	virtual int set_up()=0;
	virtual void tear_down()=0;
	virtual int init_lower_bound()=0;
	virtual Result& get_result()=0;

	
protected:
	//virtual ~ITest(){}
	//ITest(){}
	
////////////
// data members
	param_t param;
};
