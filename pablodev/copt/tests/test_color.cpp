#include "google/gtest/gtest.h"
#include "pablodev/graph/graph_gen.h"
#include "pablodev/utils/prec_timer.h"
#include "../init_color.h"

using namespace std;

template <class Collection>
void printCol(Collection& col){
	copy(col.begin(), col.end(), ostream_iterator<typename Collection::value_type>(cout, " "));
}

typedef vector<int> vint;

TEST(Color, simple_graph){
	const int SIZEG =6;
	int myints[] = {1,2,1,3,1,2};
	vint sol (myints, myints + 6);
	vint res;
	cout<<"--------------------------------------------------------"<<endl;

	//Ugraph
	ugraph ug(SIZEG);
	ug.add_edge(0, 1);
	ug.add_edge(0, 3);
	ug.add_edge(1, 2);
	ug.add_edge(1, 3);
	ug.add_edge(2, 5);

	InitColor<ugraph> c(ug);
	int col_size=c.greedyIndependentSetColoring(res);
	EXPECT_TRUE(res==sol);
		
	col_size=c.greedyColoring(res);
	EXPECT_TRUE(res==sol);

	cout<<"--------------------------------------------------------"<<endl;
}

TEST(Color_sparse, simple_graph){
	const int SIZEG =6;
	int myints[] = {1,2,1,3,1,2};
	vint sol (myints, myints + 6);
	vint res;
	cout<<"--------------------------------------------------------"<<endl;

		
	//sparse graph
	sparse_ugraph usg(SIZEG);
	usg.add_edge(0, 1);
	usg.add_edge(0, 3);
	usg.add_edge(1, 2);
	usg.add_edge(1, 3);
	usg.add_edge(2, 5);

	InitColor<sparse_ugraph> cs(usg);
	int color_size=cs.greedyIndependentSetColoring(res);
	EXPECT_TRUE(res==sol);
		
	color_size=cs.greedyColoring(res);
	EXPECT_TRUE(res==sol);

	cout<<"--------------------------------------------------------"<<endl;
}

TEST(Color_sparse, simple_subgraph){
	const int SIZEG =6;
	int myints[] = {1,2,1,3,1,2};
	vint sol (myints, myints + 6);
	vint res;
	cout<<"--------------------------------------------------------"<<endl;

		
	//sparse graph
	sparse_ugraph usg(SIZEG);
	usg.add_edge(0, 1);
	usg.add_edge(0, 3);
	usg.add_edge(1, 2);
	usg.add_edge(1, 3);
	usg.add_edge(2, 5);

	InitColor<sparse_ugraph> cs(usg);
	int color_size;

	sparse_ugraph::bb_type bbs(usg.number_of_vertices());
	bbs.set_bit(0,3);
	color_size=cs.greedyIndependentSetColoring(bbs);
	EXPECT_EQ(3, color_size);	//color: 1,2,1,3

	bbs.erase_bit(3);
	color_size=cs.greedyIndependentSetColoring(bbs);
	EXPECT_EQ(2, color_size);  //color: 1,2,1
	
	cout<<"--------------------------------------------------------"<<endl;
}

TEST(Color, brock){
	/*int myints[] = {52,59,58,55,55,51,47,57,41,56,55,1,33,32,26,10,35,25,53,54,54,40,36,49,54
		,52,45,49,53,52,51,31,51,44,42,47,50,20,45,50,39,22,46,49,37,48,41,48,28,43,47,17,44,
		26,23,34,47,46,35,45,11,43,32,10,45,38,39,44,42,38,29,41,43,42,36,35,31,19,30,28,40,41,
		19,33,33,40,39,17,15,26,12,34,37,39,38,29,37,36,22,32,9,30,27,35,28,25,27,34,33,14,32,31,
		24,5,31,30,19,29,29,28,26,27,27,2,26,24,22,25,25,22,18,15,6,23,14,8,5,23,21,3,20,24,6,20,19,
		23,21,22,21,10,20,4,19,14,18,17,18,11,16,15,17,12,13,12,16,16,7,15,14,13,9,13,12,11,11,7,4,10,
		8,1,10,7,9,4,9,6,2,8,7,6,3,5,5,3,2,4,3,2,1,1};
	std::vector<int> sol1 (myints, myints + sizeof(myints) / sizeof(int) );
	std::vector<int> res1;
	int list[200];
	cout<<"--------------------------------------------------------"<<endl;

	//-------------------------------------------------------------------------
    //Ugraph
    ugraph ug("brock200_1.clq");
	InitColor<ugraph> c(ug);
	c.init_coloring(DEFAULT_INIT_COLOR,list);
	for (int i = ug.number_of_vertices() - 1; i >= 0; i--) 
		{cout << list[i]<<" ";
		res1.push_back(list[i]);
		}
	cout<<endl;
	res1.clear();

	c.init_coloring(GREDDY,list);
	for (int i = ug.number_of_vertices() - 1; i >= 0; i--) {
		cout << list[i]<<" ";
		res1.push_back(list[i]);}
	cout<<endl;
	EXPECT_EQ(true,res1==sol1);
	res1.clear();

	cout<<"--------------------------------------------------------"<<endl;


	//-------------------------------------------------------------------------
    //Sparse Ugraph
    sparse_ugraph usg("brock200_1.clq");
	InitColor<sparse_ugraph> cs(usg);
	cs.init_coloring(DEFAULT_INIT_COLOR,list);
	for (int i = usg.number_of_vertices() - 1; i >= 0; i--) {
		cout << list[i]<<" ";
		res1.push_back(list[i]);}
	cout<<endl;
	res1.clear();

	c.init_coloring(GREDDY,list);
	for (int i = usg.number_of_vertices() - 1; i >= 0; i--){ 
		cout << list[i]<<" ";
		res1.push_back(list[i]);}
	cout<<endl;
	EXPECT_EQ(true,res1==sol1);
	res1.clear();

	c.init_coloring(TOMITA,list);
	for (int i = usg.number_of_vertices() - 1; i >= 0; i--) {
		cout << list[i]<<" ";
		res1.push_back(list[i]);}
	cout<<endl;
	EXPECT_EQ(true,res1==sol1);
	res1.clear();*/
}



TEST(Color_matrix, simple_graph){
    const int SIZEG =6;
    int myints[] = {1,2,1,3,1,2};
    vint sol (myints, myints + 6);
    vint res;
    cout<<"--------------------------------------------------------"<<endl;

    //Ugraph
    ugraph ug(SIZEG);
    ug.add_edge(0, 1);
    ug.add_edge(0, 3);
    ug.add_edge(1, 2);
    ug.add_edge(1, 3);
    ug.add_edge(2, 5);

    InitColor<ugraph> c(ug);
	c.init_ColorMatrix();
    int col_size=c.greedyColoring(res);
    EXPECT_TRUE(res==sol);    

	col_size=c.greedyColorMatrixColoring(res);
    EXPECT_TRUE(res==sol);  

    cout<<"--------------------------------------------------------"<<endl;
}

TEST(Color_matrix, correctness){
/////////////////////
// OBSERVATIONS: Currently times in this test of the two algorithms are not comparable because
// memory is allocated inside one of the algorithms

	PrecisionTimer pt;
    vint sol, res; 
    double secs, secs2, col_size,col_size2;

    const int NV_INF=2000, NV_SUP=10000, INC_SIZE=500, REP_MAX=10;
    const double DEN_INF=.1,DEN_SUP=.9, INC_DENSITY=.1;
    /*string path="auxfilerandom.txt";
    ofstream f(path, std::ofstream::out);
    f<<"tam den rep TotalGreedy Greedy TotalMatrix Matrix TotalMatrixTrans MatrixTrans"<<endl;*/
	for(int tam=NV_INF;tam<NV_SUP;tam+=INC_SIZE)  {
		for(double den=DEN_INF;den<DEN_SUP;den+=INC_DENSITY){
			for(int rep=0;rep<REP_MAX;rep++){
				cout<<"--------------------------------------------------------"<<endl;
				//-------------------------------------------------------------------------
				//Ugraph
				ugraph ug;
				RandomGen::create_ugraph(ug,tam,den);
				sol.clear();
				res.clear();

				InitColor<ugraph> c(ug);
				c.init_ColorMatrix();
				
				pt.wall_tic();
				col_size=c.greedyIndependentSetColoring(sol);
				secs=pt.wall_toc();

				pt.wall_tic();				
				col_size2=c.greedyColorMatrixColoring(res);
				secs2=pt.wall_toc();
				

				//tests for same color assignments to all vertices
				ASSERT_TRUE(res==sol);
				cout<<"[N:"<<tam<<" p:"<<den<<" r:"<<rep<<" c1:"<<col_size<<" c2:"<<col_size2<<" t1:"<<secs<<" t2:"<<secs2<<"]"<<endl;
			}
		}
	}
    //f.close();
}
