#include "benchmark_clique.h"

void BkClique::Dimacs(){
///////////////////
// BHOSH and DIMACS main set of instances for test validation (no ILS info)
// 
// Comments: Alphabetical order

	/*	add_test("brock200_1.clq");
	add_test("brock200_2.clq");
	add_test("brock200_3.clq");		
	add_test("brock200_4.clq");	
	add_test("brock400_1.clq");			
	add_test("brock400_2.clq");
	add_test("brock400_3.clq");
	add_test("brock400_4.clq");
	add_test("brock800_1.clq");			
	add_test("brock800_2.clq");				
	add_test("brock800_3.clq");				
	add_test("brock800_4.clq");

	add_test("C125.9.clq");	
	add_test("C250.9.clq");				
	//	add_test("C500.9.clq");	
	//	add_test("C1000.9.clq");	
	//	add_test("C2000.5.clq");

	add_test("dsjc500.1.clq");				
	add_test("dsjc500.5.clq");	
	//	add_test("dsjc500.9.clq");	
	add_test("dsjc1000.1.clq");	
	add_test("dsjc1000.5.clq");

	add_test("frb30-15-1.clq");	
	add_test("frb30-15-2.clq");				
	add_test("frb30-15-3.clq");	
	add_test("frb30-15-4.clq");	
	add_test("frb30-15-5.clq");

	add_test("gen200_p0.9_44.clq");	
	add_test("gen200_p0.9_55.clq");				
	add_test("gen400_p0.9_55.clq");	
	add_test("gen400_p0.9_65.clq");	
	add_test("gen400_p0.9_75.clq");*/


	add_test("hamming8-2.clq");	
	add_test("hamming8-4.clq");	
	add_test("hamming10-2.clq");

	add_test("johnson8-2-4.clq");	
	add_test("johnson8-4-4.clq");		
	add_test("johnson16-2-4.clq");
	//	add_test("johnson32-2-4.clq");

	add_test("keller4.clq");	
	add_test("keller5.clq");

	add_test("MANN_a9.clq");				
	add_test("MANN_a27.clq");				
	add_test("MANN_a45.clq");
	//add_test("MANN_a81.clq");			//parado despues de 5 dias en el I7 de casa	

	add_test("p_hat300-1.clq");				
	add_test("p_hat300-2.clq");				
	add_test("p_hat300-3.clq");				
	add_test("p_hat500-1.clq");				
	add_test("p_hat500-2.clq");				
	add_test("p_hat500-3.clq");				
	add_test("p_hat700-1.clq");				
	add_test("p_hat700-2.clq");				
	add_test("p_hat700-3.clq");				
	add_test("p_hat1000-1.clq");				
	add_test("p_hat1000-2.clq");			
	//	add_test("p_hat1000-3.clq");				
	add_test("p_hat1500-1.clq");				
	add_test("p_hat1500-2.clq");				
	//	add_test("p_hat1500-3.clq");	

	add_test("san1000.clq");	

	add_test("san200_0.7_1.clq");				
	add_test("san200_0.7_2.clq");				
	add_test("san200_0.9_1.clq");				
	add_test("san200_0.9_2.clq");			
	add_test("san200_0.9_3.clq");			
	add_test("san400_0.5_1.clq");			
	add_test("san400_0.7_1.clq");			
	add_test("san400_0.7_2.clq");			
	add_test("san400_0.7_3.clq");			
	add_test("san400_0.9_1.clq");		

	add_test("sanr200_0.7.clq");			
	add_test("sanr200_0.9.clq");				
	add_test("sanr400_0.5.clq");				
	add_test("sanr400_0.7.clq");
}

void BkClique::SubsetDimacs(){
//////////////////	
//last configuration: Review Infra-chromatic I COR (11/11/2014)

	add_test("rg_12.txt");
	add_test("rg_13.txt");
	return;
	add_test("brock200_2.clq");
	add_test("brock200_3.clq");		
	add_test("brock200_4.clq");	
	/*	add_test("brock400_1.clq");			
	add_test("brock400_2.clq");
	add_test("brock400_3.clq");
	add_test("brock400_4.clq");
	add_test("brock800_1.clq");			
	add_test("brock800_2.clq");				
	add_test("brock800_3.clq");				
	add_test("brock800_4.clq");*/

	/*add_test("c-fat200-1.clq");				
	add_test("c-fat200-2.clq");			
	add_test("c-fat200-5.clq");			
	add_test("c-fat500-1.clq");			
	add_test("c-fat500-2.clq");			 	
	add_test("c-fat500-5.clq");		
	add_test("c-fat500-10.clq");*/	

	add_test("hamming6-2.clq");				
	add_test("hamming6-4.clq");					
	add_test("hamming8-2.clq");		
	add_test("hamming8-4.clq");			      
	//	add_test("hamming10-2.clq");					
	//	add_test("hamming10-4.clq");	

	add_test("johnson8-2-4.clq");		
	add_test("johnson8-4-4.clq");		
	//	add_test("johnson16-2-4.clq");
	//	add_test("johnson32-2-4.clq");	

	/*	add_test("keller4.clq");*/				
	//	add_test("keller5.clq");				
	//	add_test("keller6.clq");	

	//	add_test("MANN_a9.clq");				
	add_test("MANN_a27.clq");				
	//	add_test("MANN_a45.clq");				
	/*	add_test("MANN_a81.clq");		*/	

	/*	add_test("san200_0.7_1.clq");				
	add_test("san200_0.7_2.clq");				
	add_test("san200_0.9_1.clq");				
	add_test("san200_0.9_2.clq");			
	add_test("san200_0.9_3.clq");			
	//	add_test("san400_0.5_1.clq");			
	add_test("san400_0.7_1.clq");			
	add_test("san400_0.7_2.clq");			
	add_test("san400_0.7_3.clq");			
	//	add_test("san400_0.9_1.clq");			
	add_test("san1000.clq");	*/

	/*	add_test("sanr200_0.7.clq");			
	add_test("sanr200_0.9.clq");				
	add_test("sanr400_0.5.clq");				
	add_test("sanr400_0.7.clq");*/

	add_test("p_hat300-1.clq");				
	add_test("p_hat300-2.clq");				
	add_test("p_hat300-3.clq");				
	add_test("p_hat500-1.clq");				
	add_test("p_hat500-2.clq");				
	add_test("p_hat500-3.clq");				
	//	add_test("p_hat700-1.clq");				
	//	add_test("p_hat700-2.clq");				
	//	add_test("p_hat700-3.clq");				
	//	add_test("p_hat1000-1.clq");				
	//	add_test("p_hat1000-2.clq");			
	//	add_test("p_hat1000-3.clq");				
	//	add_test("p_hat1500-1.clq");				
	//	add_test("p_hat1500-2.clq");				
	//	add_test("p_hat1500-3.clq");	

	//	add_test("gen200_p0.9_44.clq");	
	//	add_test("gen200_p0.9_55.clq");			
	//	add_test("gen400_p0.9_55.clq");	
	/*	add_test("gen400_p0.9_65.clq");	
	add_test("gen400_p0.9_75.clq");	*/	

	//	add_test("C125.9.clq");	
	//	add_test("C250.9.clq");				
	/*	add_test("C500.9.clq");	
	add_test("C1000.9.clq");	
	add_test("C2000.5.clq");	*/


	//add_test("dsjc250.5.clq");				
	//	add_test("dsjc500.1.clq");				
	//	add_test("dsjc500.5.clq");	
	//	add_test("dsjc500.9.clq");	
	//	add_test("dsjc1000.1.clq");	
	//	add_test("dsjc1000.5.clq");	
	//	add_test("dsjc1000.9.clq");*/

	//Small subset of Bhoshlib
	/*	add_test("frb30-15-1.clq");	
	add_test("frb30-15-2.clq");				
	add_test("frb30-15-3.clq");	
	add_test("frb30-15-4.clq");	
	add_test("frb30-15-5.clq");*/
}

void BkClique::HardDimacs(){

	add_test("p_hat1500-2.clq");
	add_test("keller5.clq");	
	add_test("gen400_p0.9_65.clq");	
	add_test("gen400_p0.9_75.clq");	
	//add_test("C500.9.clq");	

}

void BkClique::ILS(){
///////////////////
// BHOSH and DIMACS main set of instances for test validation
// Includes initial clique info as provided by strong ILS heuristic on init (Iterative Local Search heur)
//
// Comments: Alphabetical order

	add_test("brock200_1.clq",21);
	add_test("brock200_2.clq",12);
	add_test("brock200_3.clq",15);		
	add_test("brock200_4.clq",17);	
	add_test("brock400_1.clq",25);			
	add_test("brock400_2.clq",25);
	add_test("brock400_3.clq",31);
	add_test("brock400_4.clq",33);
	add_test("brock800_1.clq",21);			
	add_test("brock800_2.clq",21);				
	add_test("brock800_3.clq",22);				
	add_test("brock800_4.clq",21);

	add_test("C125.9.clq",34);	
	add_test("C250.9.clq",44);				
	//	add_test("C500.9.clq");	
	//	add_test("C1000.9.clq");	
	//	add_test("C2000.5.clq");

	add_test("dsjc500.1.clq",5);				
	add_test("dsjc500.5.clq",13);	
	//	add_test("dsjc500.9.clq");	
	add_test("dsjc1000.1.clq",6);	
	add_test("dsjc1000.5.clq",15);

	add_test("frb30-15-1.clq",30);	
	add_test("frb30-15-2.clq",30);				
	add_test("frb30-15-3.clq",30);	
	add_test("frb30-15-4.clq",30);	
	add_test("frb30-15-5.clq",30);

	add_test("gen200_p0.9_44.clq",44);	
	add_test("gen200_p0.9_55.clq",55);				
	add_test("gen400_p0.9_55.clq",55);	
	add_test("gen400_p0.9_65.clq",65);	
	add_test("gen400_p0.9_75.clq",75);


	add_test("hamming8-2.clq",128);	
	add_test("hamming8-4.clq",16);	
	add_test("hamming10-2.clq",512);

	add_test("johnson8-2-4.clq",4);	
	add_test("johnson8-4-4.clq",14);		
	add_test("johnson16-2-4.clq",8);
	//	add_test("johnson32-2-4.clq",16);

	add_test("keller4.clq",11);	
	add_test("keller5.clq",27);

	add_test("MANN_a9.clq",16);				
	add_test("MANN_a27.clq",126);				
	add_test("MANN_a45.clq",344);
	//add_test("MANN_a81.clq",1100);			//parado despues de 5 dias en el I7 de casa	

	add_test("p_hat300-1.clq",8);				
	add_test("p_hat300-2.clq",25);				
	add_test("p_hat300-3.clq",36);				
	add_test("p_hat500-1.clq",9);				
	add_test("p_hat500-2.clq",36);				
	add_test("p_hat500-3.clq",50);				
	add_test("p_hat700-1.clq",11);				
	add_test("p_hat700-2.clq",44);				
	add_test("p_hat700-3.clq",62);				
	add_test("p_hat1000-1.clq",10);				
	add_test("p_hat1000-2.clq",46);			
	//	add_test("p_hat1000-3.clq",68);				
	add_test("p_hat1500-1.clq",12);				
	add_test("p_hat1500-2.clq",65);				
	//	add_test("p_hat1500-3.clq",94);	

	add_test("san1000.clq",15);	

	add_test("san200_0.7_1.clq",30);				
	add_test("san200_0.7_2.clq",18);				
	add_test("san200_0.9_1.clq",70);				
	add_test("san200_0.9_2.clq",60);			
	add_test("san200_0.9_3.clq",44);			
	add_test("san400_0.5_1.clq",13);			
	add_test("san400_0.7_1.clq",40);			
	add_test("san400_0.7_2.clq",30);			
	add_test("san400_0.7_3.clq",22);			
	add_test("san400_0.9_1.clq",100);		

	add_test("sanr200_0.7.clq",18);			
	add_test("sanr200_0.9.clq",42);				
	add_test("sanr400_0.5.clq",13);				
	add_test("sanr400_0.7.clq",21);
}

void BkClique::EasyILS(){
////////////////////
// A subset of ILS instances for fast tests

	add_test("brock200_1.clq",21);
	add_test("brock200_2.clq",12);
	add_test("brock200_3.clq",15);		
	add_test("brock200_4.clq",17);	
	/*	add_test("brock400_1.clq",25);			
	add_test("brock400_2.clq",25);
	add_test("brock400_3.clq",31);
	add_test("brock400_4.clq",33);*/
	/*	add_test("brock800_1.clq",21);			
	add_test("brock800_2.clq",21);				
	add_test("brock800_3.clq",22);				
	add_test("brock800_4.clq",21);*/

	add_test("hamming8-2.clq",128);		
	add_test("hamming10-2.clq",512);


	add_test("johnson8-4-4.clq",14);		
	add_test("johnson16-2-4.clq",8);
	//	add_test("johnson32-2-4.clq",16);


	add_test("keller4.clq",11);	
	//	add_test("keller5.clq",27);

	add_test("MANN_a9.clq",16);				
	add_test("MANN_a27.clq",126);				
	//	add_test("MANN_a45.clq",344);
	//add_test("MANN_a81.clq",1100);			//parado despues de 5 dias en el I7 de casa	


	add_test("san200_0.7_1.clq",30);				
	add_test("san200_0.7_2.clq",18);				
	add_test("san200_0.9_1.clq",70);				
	add_test("san200_0.9_2.clq",60);			
	add_test("san200_0.9_3.clq",44);			
	add_test("san400_0.5_1.clq",13);			
	add_test("san400_0.7_1.clq",40);			
	add_test("san400_0.7_2.clq",30);			
	add_test("san400_0.7_3.clq",22);			
	add_test("san400_0.9_1.clq",100);		
	add_test("san1000.clq",15);	

	add_test("sanr200_0.7.clq",18);			
	add_test("sanr200_0.9.clq",42);				
	add_test("sanr400_0.5.clq",13);				
	add_test("sanr400_0.7.clq",21);

	add_test("p_hat300-1.clq",8);				
	add_test("p_hat300-2.clq",25);				
	add_test("p_hat300-3.clq",36);				
	add_test("p_hat500-1.clq",9);				
	add_test("p_hat500-2.clq",36);				
	add_test("p_hat500-3.clq",50);				
	add_test("p_hat700-1.clq",11);				
	add_test("p_hat700-2.clq",44);				
	//	add_test("p_hat700-3.clq",62);				
	add_test("p_hat1000-1.clq",10);				
	add_test("p_hat1000-2.clq",46);			
	//	add_test("p_hat1000-3.clq",68);				
	add_test("p_hat1500-1.clq",12);				
	//	add_test("p_hat1500-2.clq",65);				
	//	add_test("p_hat1500-3.clq",94);	


	add_test("gen200_p0.9_44.clq",44);	
	add_test("gen200_p0.9_55.clq",55);				
	/*	add_test("gen400_p0.9_55.clq",55);	
	add_test("gen400_p0.9_65.clq",65);	
	add_test("gen400_p0.9_75.clq",75);*/

	add_test("C125.9.clq",34);	
	/*	add_test("C250.9.clq",44);				
	add_test("C500.9.clq");	
	add_test("C1000.9.clq");	
	add_test("C2000.5.clq");*/

	/*add_test("frb30-15-1.clq",30);	
	add_test("frb30-15-2.clq",30);				
	add_test("frb30-15-3.clq",30);	
	add_test("frb30-15-4.clq",30);	
	add_test("frb30-15-5.clq",30);*/

	add_test("dsjc500.1.clq",5);				
	add_test("dsjc500.5.clq",13);	
	//	add_test("dsjc500.9.clq");	
	add_test("dsjc1000.1.clq",6);	
	//	add_test("dsjc1000.5.clq",15);
}

void BkClique::SubsetBhoshlib(){

	add_test("frb30-15-1.clq");	
	add_test("frb30-15-2.clq");				
	add_test("frb30-15-3.clq");	
	add_test("frb30-15-4.clq");	
	add_test("frb30-15-5.clq");

	add_test("frb35-17-1.clq");	
	add_test("frb35-17-2.clq");				
	add_test("frb35-17-3.clq");	
	add_test("frb35-17-4.clq");	
	add_test("frb35-17-5.clq");

	add_test("frb40-19-1.clq");	
	add_test("frb40-19-2.clq");				
	add_test("frb40-19-3.clq");	
	add_test("frb40-19-4.clq");	
	add_test("frb40-19-5.clq");

	add_test("frb45-21-1.clq");	
	add_test("frb45-21-2.clq");				
	add_test("frb45-21-3.clq");	
	add_test("frb45-21-4.clq");	
	add_test("frb45-21-5.clq");

	//***
}

void BkClique::Snap(){
//////////////////////////
//Dimacs format

	//snap
	add_test("0.edges");				
	add_test("1.edges");	
	add_test("2.edges");	
	add_test("3.edges");	
	add_test("4.edges");	
	add_test("5.edges");	
	add_test("6.edges");	
	add_test("7.edges");	
	add_test("8.edges");	
	add_test("9.edges");

	add_test("CA-AstroPh.txt");
	add_test("CA-CondMat.txt");


	add_test("cEmail-Enron.txt");
	add_test("com-amazon.ungraph.txt");
	add_test("com-dblp.ungraph.txt");
	add_test("com-youtube.ungraph.txt");


	add_test("oregon1_010331.txt");				
	add_test("oregon1_010407.txt");	
	add_test("oregon1_010414.txt");	
	add_test("oregon1_010421.txt");
	add_test("oregon1_010428.txt");
	add_test("oregon1_010505.txt");
	add_test("oregon1_010512.txt");
	add_test("oregon1_010519.txt");
	add_test("oregon1_010526.txt");

	//pajek
	//add_test("out.petster-carnivore");	
	add_test("out.petster-friendships-cat");	
	add_test("out.petster-friendships-dog");


	add_test("roadNet-CA.txt");	
	add_test("roadNet-PA.txt");
	add_test("roadNet-TX.txt");


	//***
}

void BkClique::Others(){
//For ad hoc tests

	add_test("p_hat1500-2.clq");	
	add_test("san200_0.9_3.clq");	
	add_test("san400_0.7_2.clq");	
	add_test("gen400_p0.9_65.clq");	
	add_test("gen400_p0.9_75.clq");
	add_test("keller5.clq");
}

