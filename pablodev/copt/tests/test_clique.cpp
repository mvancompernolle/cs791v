
#include "google/gtest/gtest.h"
#include "pablodev/graph/graph_gen.h"
#include "../clique.h"

using namespace std;
//
//TEST(Clique, basic){
//	cout<<"--------------------------------------------------------"<<endl;
//	//-------------------------------------------------------------------------
//	//Ugraph
//	ugraph ug(6);
//	ug.add_edge(0, 1);
//	ug.add_edge(0, 3);
//	ug.add_edge(1, 2);
//	ug.add_edge(1, 3);
//	ug.add_edge(2, 5);
//	Clique<ugraph> cug(ug);
//	cug.set_up();
//	cug.run_unrolled();
//	EXPECT_EQ(3,cug.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//
//	//-------------------------------------------------------------------------
//	//Sparse Ugraph
//	sparse_ugraph usg(6);
//	usg.add_edge(0, 1);
//	usg.add_edge(0, 3);
//	usg.add_edge(1, 2);
//	usg.add_edge(1, 3);
//	usg.add_edge(2, 5);
//	Clique<sparse_ugraph> cusg(usg);
//	cusg.set_up();
//	cusg.run_unrolled();
//	EXPECT_EQ(3,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//	
//}


//TEST(Clique, brock_200_1_unrolled){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("brock200_1.clq");
//	Clique<sparse_ugraph> cusg(usg);
//	int res= cusg.set_up_unrolled();
//	cout<<"finished setup unrolled"<<endl;
//
//	if(res==0){
//		cusg.run_unrolled();
//	}
//		
//	EXPECT_EQ(21,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//}
//
//TEST(Clique, brock_200_1){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("brock200_1.clq");
//	Clique<sparse_ugraph> cusg(usg);
//	int res=cusg.set_up(); 
//	cout<<"finished setup"<<endl;
//
//	if(res==0){
//		cusg.run();
//	}
//		
//	EXPECT_EQ(21,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//}

//TEST(Clique, real_roadNet_PA){
//
//	cout<<"--------------------------------------------------------"<<endl;
//	
//	////Ugraph
//	//ugraph ug("roadNet-PA.txt");
//	//Clique<ugraph> cug(ug);
//	//if(cug.set_up()>0){
//	//	return;
//	//}
//	////cug.run();
//	//cug.run_unrolled();
//	//EXPECT_EQ(26,cug.get_max_clique());
//	//cout<<"--------------------------------------------------------"<<endl;
//		
//    //Sparse Ugraph
// //   sparse_ugraph usg("roadNet-PA.txt");
//	//Clique<sparse_ugraph> cusg(usg);
//	//if(cusg.set_up()>0){
//	//	return;
//	//}
//	//cout<<"finished setup"<<endl;
//	////cusg.run();
//	//cusg.run_unrolled();
//	//EXPECT_EQ(4,cusg.get_max_clique());
//	//cout<<"--------------------------------------------------------"<<endl;
//}
//

//TEST(Clique, real_CA_CondMat_unrolled){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("ca-CondMat.mtx");
//	Clique<sparse_ugraph> cusg(usg);
//	int res= cusg.set_up_unrolled();
//	cout<<"finished setup unrolled"<<endl;
//
//	if(res==0){
//		cusg.run_unrolled();
//	}
//		
//	EXPECT_EQ(26,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//	
//}
//
//TEST(Clique, real_CA_CondMat){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("ca-CondMat.mtx");
//	Clique<sparse_ugraph> cusg(usg);
//	int res=cusg.set_up(); 
//	cout<<"finished setup"<<endl;
//
//	if(res==0){
//		cusg.run();
//	}
//		
//	EXPECT_EQ(26,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//}
//
//TEST(Clique, real_you_tube_snap_unrolled){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("soc-youtube-snap.mtx");
//	Clique<sparse_ugraph> cusg(usg);
//	int res= cusg.set_up_unrolled();
//	cout<<"finished setup unrolled"<<endl;
//
//	if(res==0){
//		cusg.run_unrolled();
//	}
//		
//	EXPECT_EQ(17,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//	
//}
//
//TEST(Clique, real_you_tube_snap){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("soc-youtube-snap.mtx");
//	Clique<sparse_ugraph> cusg(usg);
//	int res=cusg.set_up(); 
//	cout<<"finished setup"<<endl;
//
//	if(res==0){
//		cusg.run();
//	}
//		
//	EXPECT_EQ(17,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//}
//
//TEST(Clique, real_tech_as_skitter_unrolled){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("tech-as-skitter.mtx");
//	Clique<sparse_ugraph> cusg(usg);
//	int res= cusg.set_up_unrolled();
//	cout<<"finished setup unrolled"<<endl;
//
//	if(res==0){
//		cusg.run_unrolled();
//	}
//		
//	EXPECT_EQ(67,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//	
//}
//
//TEST(Clique, real_tech_as_skitter){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("tech-as-skitter.mtx");
//	Clique<sparse_ugraph> cusg(usg);
//	int res=cusg.set_up(); 
//	cout<<"finished setup"<<endl;
//
//	if(res==0){
//		cusg.run();
//	}
//		
//	EXPECT_EQ(67,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//}
//
//TEST(Clique, fe_rotor_unrolled){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("fe_rotor.mtx");
//	Clique<sparse_ugraph> cusg(usg);
//	int res= cusg.set_up_unrolled();
//	cout<<"finished setup unrolled"<<endl;
//
//	if(res==0){
//		cusg.run_unrolled();
//	}
//		
//	EXPECT_EQ(5,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//	
//}

//TEST(Clique, fe_rotor){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("fe_rotor.mtx");
//	Clique<sparse_ugraph> cusg(usg);
//	int res=cusg.set_up(); 
//	cout<<"finished setup"<<endl;
//
//	if(res==0){
//		cusg.run();
//	}
//		
//	EXPECT_EQ(5,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//}
//
//
//TEST(Clique, delaunay_n19_unrolled){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("delaunay_n20.mtx");
//	usg.print_data();
//	Clique<sparse_ugraph> cusg(usg);
//	int res= cusg.set_up_unrolled();
//	cout<<"finished setup unrolled"<<endl;
//
//	if(res==0){
//		cusg.run_unrolled();
//	}
//		
//	EXPECT_EQ(4,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//	
//}
//
//TEST(Clique, delaunay_n19_unrolled_uncolored){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("delaunay_n20.mtx");
//	usg.print_data();
//	Clique<sparse_ugraph> cusg(usg);
//	int res= cusg.set_up_unrolled();
//	cout<<"finished setup unrolled"<<endl;
//
//	if(res==0){
//		cusg.run_unrolled_without_coloring();
//	}
//		
//	EXPECT_EQ(4,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//	
//}
//
//TEST(Clique_sparse, delaunay_n19_uncolored){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("delaunay_n20.mtx");
//	usg.print_data();
//	Clique<sparse_ugraph> cusg(usg);
//	int res=cusg.set_up(); 
//	cout<<"finished setup"<<endl;
//
//	if(res==0){
//		cusg.run_without_coloring();
//	}
//		
//	EXPECT_EQ(4,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//}
//
//TEST(Clique_sparse, delaunay_n19){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("delaunay_n20.mtx");
//	usg.print_data();
//	Clique<sparse_ugraph> cusg(usg);
//	int res=cusg.set_up(); 
//	cout<<"finished setup"<<endl;
//
//	if(res==0){
//		cusg.run();
//	}
//		
//	EXPECT_EQ(4,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//}
////
//TEST(Clique, sc_ldoor_unrolled){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("sc-ldoor.mtx");
//	usg.print_data();
//	Clique<sparse_ugraph> cusg(usg);
//	int res= cusg.set_up_unrolled();
//
//	if(res==0){
//		cusg.run_unrolled(true);
//	}
//		
//	EXPECT_EQ(21,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//	
//}
//
//TEST(Clique,  sc_ldoor_unrolled_uncolored){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("sc-ldoor.mtx");
//	usg.print_data();
//	Clique<sparse_ugraph> cusg(usg);
//	int res= cusg.set_up_unrolled();
//	cout<<"finished setup unrolled"<<endl;
//
//	if(res==0){
//		cusg.run_unrolled_without_coloring();
//	}
//		
//	EXPECT_EQ(21,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//	
//}
//
////TEST(Clique_sparse,  sc_ldoor_uncolored){
////////////////////
//////  Sparse Ugraph
////
////	cout<<"--------------------------------------------------------"<<endl;
////    sparse_ugraph usg("sc-ldoor.mtx");
////	usg.print_data();
////	Clique<sparse_ugraph> cusg(usg);
////	int res=cusg.set_up(); 
////	cout<<"finished setup"<<endl;
////
////	if(res==0){
////		cusg.run_without_coloring();
////	}
////		
////	EXPECT_EQ(21,cusg.get_max_clique());
////	cout<<"--------------------------------------------------------"<<endl;
////}
//
//TEST(Clique_sparse,  sc_ldoor){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("sc-ldoor.mtx");
//	usg.print_data();
//	Clique<sparse_ugraph> cusg(usg);
//	int res=cusg.set_up(); 
//	cout<<"finished setup"<<endl;
//
//	if(res==0){
//		cusg.run();
//	}
//		
//	EXPECT_EQ(21,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//}
//
//TEST(Clique, hugetrace_0000_unrolled){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("hugetrace-00000.mtx");
//	usg.print_data();
//	Clique<sparse_ugraph> cusg(usg);
//	int res= cusg.set_up_unrolled();
//	cout<<"finished setup unrolled"<<endl;
//
//	if(res==0){
//		cusg.run_unrolled();
//	}
//		
//	EXPECT_EQ(2,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//	
//}
//
//TEST(Clique,  hugetrace_0000_unrolled_uncolored){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("hugetrace-00000.mtx");
//	usg.print_data();
//	Clique<sparse_ugraph> cusg(usg);
//	int res= cusg.set_up_unrolled();
//	cout<<"finished setup unrolled"<<endl;
//
//	if(res==0){
//		cusg.run_unrolled_without_coloring();
//	}
//		
//	EXPECT_EQ(2,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//	
//}
//
////TEST(Clique_sparse,  hugetrace_0000_uncolored){
////////////////////
//////  Sparse Ugraph
////
////	cout<<"--------------------------------------------------------"<<endl;
////    sparse_ugraph usg("hugetrace-00000.mtx");
////	usg.print_data();
////	Clique<sparse_ugraph> cusg(usg);
////	int res=cusg.set_up(); 
////	cout<<"finished setup"<<endl;
////
////	if(res==0){
////		cusg.run_without_coloring();
////	}
////		
////	EXPECT_EQ(2,cusg.get_max_clique());
////	cout<<"--------------------------------------------------------"<<endl;
////}
//
//TEST(Clique_sparse, hugetrace_0000){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("hugetrace-00000.mtx");
//	usg.print_data();
//	Clique<sparse_ugraph> cusg(usg);
//	int res=cusg.set_up(); 
//	cout<<"finished setup"<<endl;
//
//	if(res==0){
//		cusg.run();
//	}
//		
//	EXPECT_EQ(2,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//}
//

//TEST(Clique_sparse, some_graph){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("auxfilerandom.txt");
//	Clique<sparse_ugraph> cusg(usg);
//	int res= cusg.set_up_unrolled();
//	cout<<"finished setup unrolled"<<endl;
//	
//	if(res==0){
//		cusg.run_unrolled();
//	}
//		
//	EXPECT_EQ(3,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//}
//
//TEST(Clique, some_graph){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    ugraph usg("auxfilerandom.txt");
//	Clique<ugraph> cusg(usg);
//	int res= cusg.set_up();
//	cout<<"finished setup unrolled"<<endl;
//
//	if(res==0){
//		cusg.run();
//	}
//		
//	EXPECT_EQ(3,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//}

//TEST(Clique_sparse,EDGES_format){
//////////////////
////  Sparse Ugraph
//
//	cout<<"--------------------------------------------------------"<<endl;
//    sparse_ugraph usg("bio-yeast-protein-inter.edges");
//	// sparse_ugraph usg("soc-orkut-dir.edges");
//	Clique<sparse_ugraph> cusg(usg);
//	int res= cusg.set_up_unrolled();
//	cout<<"finished setup unrolled"<<endl;
//	
//	if(res==0){
//		cusg.run_unrolled();
//	}
//		
//	EXPECT_EQ(6,cusg.get_max_clique());
//	cout<<"--------------------------------------------------------"<<endl;
//}


TEST(Clique_Rand, basic){
/////////////////
// Compares clique number of sparse_ugraph and ugraph for a set of randomly generated graphs
// author: alopez
//
// REMARKS: used intermediate file to communicate between both types

    const int TAM_INF=300,TAM_SUP=1000, INC_SIZE=50, REP_MAX=500;
    const double DEN_INF=.02,DEN_SUP=.2, INC_DENSITY=.01;
    string path="auxfilerandom.txt";

    for(int tam=TAM_INF;tam<TAM_SUP;tam+=INC_SIZE)  {
        for(double den=DEN_INF;den<DEN_SUP;den+=INC_DENSITY){
            for(int rep=0;rep<REP_MAX;rep++){
                cout<<"--------------------------------------------------------"<<endl;
                //-------------------------------------------------------------------------
                //Ugraph
                ugraph ug;
                RandomGen::create_ugraph(ug,tam,den);
                ofstream f(path, std::ofstream::out);   
                ug.write_dimacs(f);
                f.close();
                Clique<ugraph> cug(ug);
                if(cug.set_up()==0);
							 cug.run();
    
                //-------------------------------------------------------------------------
                //Sparse Ugraph
                SparseRandomGen<> spgen;    
                sparse_ugraph usg(path);
                Clique<sparse_ugraph> cusg(usg);
                if(cusg.set_up_unrolled()==0)
							 cusg.run_unrolled();
                //remove("auxfilerandom.txt"); //Puede quitarse porque machaca el archivo
                ASSERT_EQ(cug.get_max_clique(),cusg.get_max_clique());
                cout<<"--------------------------------------------------------"<<tam<<" "<<den<<" "<<rep<<endl;
            }
        }
    }
}
