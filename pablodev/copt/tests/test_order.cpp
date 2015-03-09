#include "google/gtest/gtest.h"
#include "pablodev/graph/graph_gen.h"
#include "../init_order.h"

using namespace std;
TEST(Order, support){
	cout<<"--------------------------------------------------------"<<endl;

	//-------------------------------------------------------------------------
	//Ugraph
	ugraph ug(100);
	InitOrder<ugraph> o(ug);
	ug.add_edge(0, 1);
	ug.add_edge(1, 2);
	ug.add_edge(2, 80);
	EXPECT_EQ(3,o.deg_of_neighbors(1));
	cout<<"Test support ugraph passed"<<endl;
	cout<<"--------------------------------------------------------"<<endl;


	
	////-------------------------------------------------------------------------
	////Sparse Ugraph
	//sparse_ugraph usg(100);
	//usg.add_edge(0, 1);
	//usg.add_edge(1, 2);
	//usg.add_edge(2, 80);
	//InitOrder<sparse_ugraph> os(usg);
	//EXPECT_EQ(3,os.deg_of_neighbors(1));
	//cout<<"Test support sparse ugraph passed"<<endl;
	//cout<<"--------------------------------------------------------"<<endl;
	
}
TEST(Order, reorder_simple){
	
	
	int myints[] = {5,4,2,3,1,0};
	std::vector<int> sol1 (myints, myints + sizeof(myints) / sizeof(int) );
	int myints2[] = {5,3,1,2,4,0};
	std::vector<int> sol2 (myints2, myints2 + sizeof(myints2) / sizeof(int) );
	cout<<"--------------------------------------------------------"<<endl;


	//-------------------------------------------------------------------------
	//Ugraph
	ugraph ug(76);
	InitOrder<ugraph> o(ug);
	ug.add_edge(1, 2);
	ug.add_edge(2, 3);
	ug.add_edge(2, 5);
	ug.add_edge(4, 5);

	ug.print_edges();
	//InitOrder<>::print(o.create_new_order(MIN_WIDTH));
	//EXPECT_EQ(true,o.create_new_order(MIN_WIDTH)==sol1);
	//cout<<"initial order MIN WIDTH for ugraph passed"<<endl;
	/*InitOrder<>::print(o.create_new_order(MIN_WIDTH_MIN_TIE_STATIC));
	EXPECT_EQ(true,o.create_new_order(MIN_WIDTH_MIN_TIE_STATIC)==sol2);
	o.reorder(o.create_new_order(MIN_WIDTH_MIN_TIE_STATIC));*/
	InitOrder<>::print(o.create_new_order(NONE));
	o.reorder(o.create_new_order(NONE));
	ug.print_edges();
//	cout<<"initial order MIN WIDTH MIN TIE STATIC for ugraph passed"<<endl;
	cout<<"--------------------------------------------------------"<<endl;

	
	//-------------------------------------------------------------------------
	//Sparse Ugraph
	//sparse_ugraph usg(76);
	//InitOrder<sparse_ugraph> os(usg);	
	//usg.add_edge(1, 2);
	//usg.add_edge(2, 3);
	//usg.add_edge(2, 5);
	//usg.add_edge(4, 5);
	//

	//usg.print_edges();
	////InitOrder<>::print(o.create_new_order(MIN_WIDTH));
	////EXPECT_EQ(true,o.create_new_order(MIN_WIDTH)==sol1);
	////cout<<"initial order MIN WIDTH for ugraph passed"<<endl;
	///*InitOrder<>::print(o.create_new_order(MIN_WIDTH_MIN_TIE_STATIC));
	//EXPECT_EQ(true,o.create_new_order(MIN_WIDTH_MIN_TIE_STATIC)==sol2);
	//o.reorder(o.create_new_order(MIN_WIDTH_MIN_TIE_STATIC));*/
	//InitOrder<>::print(os.create_new_order(NONE));
	//os.reorder(os.create_new_order(NONE));
	//usg.print_edges();
	cout<<"--------------------------------------------------------"<<endl;

}

TEST(Order, reorder_Brock){
    int myints[] = {176,126,41,15,148,152,80,25,175,110,106,198,59,134,30,58,167,96,98,20,158,76,154,118,129,62,14,112,95,
        181,162,24,171,23,144,40,138,111,179,142,186,33,105,79,84,64,157,29,67,13,125,165,61,192,115,121,94,63,32,178,164,
        104,151,137,117,185,187,170,101,44,191,196,22,133,91,180,87,93,19,86,109,21,12,83,182,120,75,173,194,128,174,163,
        124,54,90,50,57,108,97,18,28,177,31,53,100,147,38,49,70,132,60,150,197,27,17,11,92,141,82,48,146,43,189,89,99,42,114,
        74,16,127,52,39,172,193,140,184,37,36,26,119,149,130,47,35,161,199,143,190,10,169,183,9,168,123,122,34,73,85,8,81,107,
        72,71,7,6,136,160,88,153,113,69,5,166,102,4,156,145,78,77,3,66,116,139,155,103,68,131,51,2,1,56,135,0,55,46,195,45,188,65,159};
    std::vector<int> sol1 (myints, myints + sizeof(myints) / sizeof(int) );
    int myints2[] = {172,126,77,8,161,155,163,19,168,115,61,198,64,136,116,68,167,72,114,43,169,78,157,129,90,13,25,26,98,140,
        154,1,158,47,148,18,137,92,173,139,185,37,75,91,82,51,152,16,96,93,133,143,86,190,107,121,102,14,6,176,76,118,151,67,110,
        184,187,179,89,111,191,195,45,131,44,177,100,95,7,71,113,59,24,58,182,103,99,174,194,119,175,142,63,65,85,62,66,117,41,30,
        80,180,0,2,120,146,29,53,9,145,105,162,197,23,42,38,31,135,40,69,149,11,188,178,32,87,134,49,50,123,20,15,164,193,141,186,
        127,12,84,122,150,124,60,4,153,199,156,189,22,166,183,101,181,128,125,81,39,55,36,54,132,171,35,52,33,112,170,74,108,104,83,
        34,144,88,3,28,147,97,57,17,70,109,138,159,56,21,130,94,106,5,48,160,27,73,46,196,79,192,10,165};
    std::vector<int> sol2 (myints2, myints2 + sizeof(myints2) / sizeof(int) );
    cout<<"--------------------------------------------------------"<<endl;


    //-------------------------------------------------------------------------
    //Ugraph
    ugraph ug("brock200_1.clq");
    InitOrder<ugraph> o(ug);
    InitOrder<>::print(o.create_new_order(MIN_WIDTH));
    EXPECT_EQ(true,o.create_new_order(MIN_WIDTH)==sol1);
    cout<<"initial order MIN WIDTH for ugraph passed"<<endl;
    InitOrder<>::print(o.create_new_order(MIN_WIDTH_MIN_TIE_STATIC));
    EXPECT_EQ(true,o.create_new_order(MIN_WIDTH_MIN_TIE_STATIC)==sol2);
    o.reorder(o.create_new_order(MIN_WIDTH_MIN_TIE_STATIC));
    cout<<"initial order MIN WIDTH MIN TIE STATIC for ugraph passed"<<endl;
    cout<<"--------------------------------------------------------"<<endl;



    //-------------------------------------------------------------------------
    //Sparse Ugraph
  /*  sparse_ugraph usg("brock200_1.clq");
    InitOrder<sparse_ugraph> os(usg);
    InitOrder<>::print(os.create_new_order(MIN_WIDTH));
    EXPECT_EQ(true,os.create_new_order(MIN_WIDTH)==sol1);
    cout<<"initial order MIN WIDTH for sparse ugraph passed"<<endl;
    InitOrder<>::print(os.create_new_order(MIN_WIDTH_MIN_TIE_STATIC));
    EXPECT_EQ(true,os.create_new_order(MIN_WIDTH_MIN_TIE_STATIC)==sol2);
    cout<<"initial order MIN WIDTH MIN TIE STATIC for sparse ugraph passed"<<endl;
    os.reorder(os.create_new_order(MIN_WIDTH_MIN_TIE_STATIC));*/
    cout<<"--------------------------------------------------------"<<endl;
}

TEST(Order_in_place, reorder_simple){
        
    int myints[] = {5,4,2,3,1,0};
    std::vector<int> sol1 (myints, myints + (sizeof(myints) / sizeof(int)) );
    int myints2[] = {5,3,1,2,4,0};
    std::vector<int> sol2 (myints2, myints2 + (sizeof(myints2) / sizeof(int)) );
    cout<<"--------------------------------------------------------"<<endl;
	
    //-------------------------------------------------------------------------
    //Ugraphs
    sparse_ugraph ug(106);    
    InitOrder<sparse_ugraph> o(ug);
    ug.add_edge(3, 1);
    ug.add_edge(3, 2);
    ug.add_edge(3, 4);
    ug.add_edge(3, 5);
    ug.add_edge(2, 5);
    ug.add_edge(4, 5);
    ug.add_edge(78, 5);

    sparse_ugraph ug2(106);
    InitOrder<sparse_ugraph> o2(ug2);
    ug2.add_edge(3, 1);
    ug2.add_edge(3, 2);
    ug2.add_edge(3, 4);
    ug2.add_edge(3, 5);
    ug2.add_edge(2, 5);
    ug2.add_edge(4, 5);
    ug2.add_edge(78, 5);
    cout<<"1:"<<endl;
    ug.print_edges();
    cout<<"2:"<<endl;
    ug2.print_edges();
    cout<<"--------------"<<endl;
   
	//checks reverse order
    InitOrder<>::print(o.create_new_order(NONE, PLACE_LF));		
    o.reorder(o.create_new_order(NONE));
    o2.reorder_in_place(o2.create_new_order(NONE));
    cout<<"1:"<<endl;
    ug.print_edges();
    cout<<"2:"<<endl;
    ug2.print_edges();
    cout<<"--------------"<<endl;
   
	//Comparison between both graphs edge by edge
	//(currently operator == between graphs is not working because equality between sparse_bitarrays does not check empty bitblocks)
    int j,j2;
    ASSERT_TRUE(ug.number_of_vertices()==ug2.number_of_vertices());
    for(int i=0;i<ug.number_of_vertices();i++){
        sparse_bitarray neigh=ug.get_neighbors(i);
		sparse_bitarray neigh2=ug2.get_neighbors(i);
        if((neigh.init_scan(bbo::NON_DESTRUCTIVE)!=EMPTY_ELEM) && (neigh2.init_scan(bbo::NON_DESTRUCTIVE)!=EMPTY_ELEM)) {
            while(true){
                j=neigh.next_bit();
                j2=neigh2.next_bit();
                ASSERT_TRUE(j==j2);
                if(j==EMPTY_ELEM)
                        break;                
            }
        }
    }

    cout<<"--------------------------------------------------------"<<endl;
}

TEST(Order_in_place, correctness){
///////////
// Random tests: at the moment NOT CORRECT

    PrecisionTimer pt;
    vint sol, res; 
    double secs, secs2, col_size,col_size2;
	int j,j2;

    const int NV_INF=1000, NV_SUP=5000, INC_SIZE=500, REP_MAX=10;
    const double DEN_INF=.1,DEN_SUP=.9, INC_DENSITY=.1;
    for(int tam=NV_INF;tam<NV_SUP;tam+=INC_SIZE)  {
        for(double den=DEN_INF;den<DEN_SUP;den+=INC_DENSITY){
            for(int rep=0;rep<REP_MAX;rep++){
                cout<<"--------------------------------------------------------"<<endl;
               
                //Ugraph
                sparse_ugraph ug;
                SparseRandomGen<>().create_ugraph(ug,tam,den);
				sparse_ugraph ug2(ug);
				
				InitOrder<sparse_ugraph> o(ug);
				InitOrder<sparse_ugraph> o2(ug2);
				

				//Reverse ordering
                o.reorder(o.create_new_order(NONE, PLACE_LF));
				o2.reorder_in_place(o2.create_new_order(NONE, PLACE_LF));
			
                //Test edge by edge (should be changed to == operator when it works for sparse graphs)
                ASSERT_TRUE(ug.number_of_vertices()==ug2.number_of_vertices());
				for(int i=0;i<ug.number_of_vertices();i++){
                    sparse_bitarray neigh=ug.get_neighbors(i);
                    sparse_bitarray neigh2=ug2.get_neighbors(i);
                    if((neigh.init_scan(bbo::NON_DESTRUCTIVE)!=EMPTY_ELEM) && (neigh2.init_scan(bbo::NON_DESTRUCTIVE)!=EMPTY_ELEM)) {
                        while(true){
                            j=neigh.next_bit();
                            j2=neigh2.next_bit();
							if(j!=j2){
								neigh.print(); cout<<endl;
								neigh2.print(); cout<<endl;
								ug.get_neighbors(j).print(); cout<<endl;
								ug2.get_neighbors(j).print(); cout<<endl;
								cout<<"i:"<<i<<" j:"<<j<<endl;
							}
                            ASSERT_TRUE(j==j2);
                            if(j==EMPTY_ELEM)
                                    break;                
                        }
                    }
                }
                        
          
				//Reorders using MIN_WIDTH
                o.reorder(o.create_new_order(MIN_WIDTH));
                o2.reorder_in_place(o2.create_new_order(MIN_WIDTH));

                //Test edge by edge (should be changed to == operator when it works for sparse graphs)
                ASSERT_TRUE(ug.number_of_vertices()==ug2.number_of_vertices());
			    for(int i=0;i<ug.number_of_vertices();i++){
                    sparse_bitarray neigh=ug.get_neighbors(i);
                    sparse_bitarray neigh2=ug2.get_neighbors(i);
                    if((neigh.init_scan(bbo::NON_DESTRUCTIVE)!=EMPTY_ELEM) && (neigh2.init_scan(bbo::NON_DESTRUCTIVE)!=EMPTY_ELEM)) {
                        while(true){
                            j=neigh.next_bit();
                            j2=neigh2.next_bit();
                            ASSERT_TRUE(j==j2);
                            if(j==EMPTY_ELEM)
                                    break;                
                        }
                    }

                }

                //tests for same color assignments to all vertices
                cout<<"[N:"<<tam<<" p:"<<den<<" r:"<<rep<<"]"<<endl;
            }
        }
    }
}


