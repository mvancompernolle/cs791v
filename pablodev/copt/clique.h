//init_color.h: header for InitColor implementation, a wrapper to color bit encoded
//				undirected graphs using greedy independent set heuristic
//
//date of creation: 23/09/14
//last update: 23/09/14
//author: pablo san segundo

#ifndef __CLIQUE_H__
#define __CLIQUE_H__

#include "clique_types.h"
#include "pablodev/graph/graph.h"
#include "pablodev/graph/kcore.h"
#include "init_color.h"
#include "init_order.h"
#include "pablodev/utils/prec_timer.h"				
#include <map>

class ICliqueAlg{
public:
	void set_param(const param_t& p){param=p;}
	param_t get_param(){return param;}

	//virtual Result run()=0;
	//virtual Result& get_result()=0;
	virtual void run(bool)=0;	
	virtual int set_up()=0;
	virtual void tear_down()=0;
	
protected:

////////////
// data members
	param_t param;
};


template <class T>
class Clique:public ICliqueAlg{
//an array of nodes (more efficient than STL implementations)
struct lista_nodos_t{
	lista_nodos_t():nodos(NULL),index(0){}
	int* nodos;
	int index;
};

typedef map<int, int> mint_t;

public:
////////////////
// constructors/destructors
	Clique(T& gout):g(gout)	{	
		m_bbnodos=NULL;
		m_lnodos=NULL;	
		m_path=NULL;										
		m_lcol_labels=NULL;
		maxac=0;
		maxno=0;
		m_alloc=g.max_degree_of_graph();
		m_size=g.number_of_vertices();
	}
	~Clique(){clear_search_basics(); clear_color_labels();}
		
////////////////
// setters and getters
	int get_max_clique()	{return maxno;}
	void set_malloc(int M)	{m_alloc=M;}
	void set_initial_lb(int lb)	{maxno=lb;}
	mint_t& get_filter()	{return m_filter;}

/////////////////////////////////
//init search data structures
	int init_search_basics();
	void clear_search_basics();

	int init_color_labels();
	void clear_color_labels();

	int init_aux_data();

//////////////////
//approximate vertex coloring
	inline void paint				(int depth);					//sequential greedy independent set coloring procedure

//////////////////
// search 
	int set_up();
	int set_up_unrolled					();
	void tear_down						() {clear_search_basics(); clear_color_labels(); m_alloc=0;}
	void run							(bool info=false);
	void run_without_coloring			(bool info=false);
	void run_unrolled					(bool info=false);
	void run_unrolled_without_coloring	(bool info=false);

	void expand								(int depth, typename T::bb_type& l_bb , lista_nodos_t& l_v);	//recursive search procedure
	void expand_without_coloring			 (int maxac, typename T::bb_type& l_bb);
	void initial_expand						();	
	void initial_expand_without_coloring	(int maxac, typename T::bb_type& l_bb);
//////////////////
// heuristics
	void filter_heur				(int maxno, typename T::bb_type& l_bb , lista_nodos_t& l_v);			//kcore filter
	void filter_heur_shrink			(int maxno);															//kcore filter

///////////////////
// I/O
	void print_filter				(ostream& o=cout);

private:	
	T& g;															//T restricted to ugraph and sparse_ugraph
	
	typename T::bb_type*	m_bbnodos;								//[DEPTH][MAX_VERTEX]
	typename T::bb_type		m_bbroot;								//bitstring at root node

	typename T::bb_type		m_sel;
	typename T::bb_type		m_unsel;

	int	**					m_lcol_labels;							//[DEPTH][MAX_VERTEX]
	lista_nodos_t*			m_lnodos;								//[DEPTH]
	lista_nodos_t			 m_lroot;								//initial list of nodes at root
	int*					m_path;									//[DEPTH]
	
	mint_t					m_filter;								//[MAXNO]-->[FIRST_VERTEX_PRUNED]

	int maxno;														//size of largest clique found in current branch 
	int maxac;														//size of current best clique found at any moment
	int m_alloc;
	int m_size;
};

///////////////
// notation
#define  LISTA_L(depth)		m_lnodos[(depth)]						//conventional list of vertices
#define  LISTA_BB(depth)    m_bbnodos[(depth)]						//list of vertices encoded as a bitstring
#define	 MAXAC_PLUS1		maxac+1
#define  MAXINT				0x1FFFFFFF								//my own MAX_INT

#include "clique_hidden.h"											//ony for developpers: client users please remove!

template<class T>
void Clique<T>::clear_search_basics (){
	if(m_bbnodos!=NULL){
		delete [] m_bbnodos;  //Llama a destructores de BBN
	}

	m_bbnodos=NULL;

	//lista de nodos
	if(m_lnodos!=NULL){
		for(int i=0; i<m_alloc; i++){
			if(m_lnodos[i].nodos!=NULL && m_lnodos[i].index==0){
				delete [] m_lnodos[i].nodos;
			}
			m_lnodos[i].nodos=NULL;
		}
	delete [] m_lnodos;		//Llama a destructores de STL::Vector
	}
	m_lnodos=NULL;

	//list at root
	if(m_lroot.nodos!=NULL){
		delete [] m_lroot.nodos;
	}
	m_lroot.nodos=NULL;
	m_lroot.index=0;


	//path
	if(m_path!=NULL)
		delete [] m_path;
	m_path=NULL;
}

template<class T>
int Clique<T>::init_search_basics (){

	m_bbnodos=new typename T::bb_type[m_alloc];
	for(int i=0; i<m_alloc; i++){
		m_bbnodos[i].init(m_size);			//set_to_0
	}

	//bitstring at root node
	m_bbroot.init(m_size);
	m_bbroot.set_bit(0,m_size-1);

	//list of nodes
	m_lnodos=new lista_nodos_t[m_alloc];
	for(int i=0; i<m_alloc; i++){
		m_lnodos[i].nodos=new int [m_size];				//index=0 en el constructor
	}
	
	//list of nodes at root
	m_lroot.nodos=new int [m_size];
	for(int i=0; i<m_size; i++){
		m_lroot.nodos[i]=i;
	}
	m_lroot.index=m_size-1;								    //vertices read from first to last

		
	//path
	m_path= new int[m_size];
	for(int i=0; i<m_size; i++)
				m_path[i]=EMPTY_ELEM;
	return 0;	
}

template<class T>
void Clique<T>::clear_color_labels(){
	if(m_lcol_labels!=NULL){
		for(int i=0; i<m_alloc; i++)	
			 delete [] m_lcol_labels[i]; 
	
	delete [] m_lcol_labels;  
	}
	m_lcol_labels=NULL;
}

template<class T>
int Clique<T>::init_color_labels(){
//////////////////
// 	
	clear_color_labels();
	m_lcol_labels=new int* [m_alloc];				

	for(int i=0; i<m_alloc;i++){
		m_lcol_labels[i]=new int [m_size];					
		for(int j=0; j<m_size; j++){
#ifdef _WIN32
		m_lcol_labels[i][j]=MAXINT;		/*before EMPTY_ELEM but it is better to give a real upper threshold*/
#else
		m_lcol_labels[i][j]=MAXINT;
#endif
		} 
	}
	return 0;
}

template<class T>
int Clique<T>::init_aux_data (){

	m_unsel.init(m_size);		
	m_sel.init(m_size);

	return 0;
}

template<class T>
inline void Clique<T>::paint (int depth){
///////////////////
// Sequential greedy independent set vertex coloring which prunes the search tree

	int col=1, kmin=maxno-depth, nBB=EMPTY_ELEM, v=EMPTY_ELEM;		
	LISTA_L(depth).index=EMPTY_ELEM;											//cleans the set fo candidate vertices
	const int DEPTH_PLUS1=depth+1;
	
	//copies list of vertices to color and stores size for fast empty check 
	int pc= (m_unsel=LISTA_BB(depth)).popcn64();
	
	//CUT based on population size
	if(pc<kmin){
			return;
	}

	while(true){ 
		m_sel=m_unsel;
		m_sel.init_scan(bbo::DESTRUCTIVE);
		while(true){
			v=m_sel.next_bit_del(nBB,m_unsel);
			if(v==EMPTY_ELEM)
							break;
			if(col>=kmin){  
				LISTA_L(depth).nodos[++LISTA_L(depth).index]=v;
				m_lcol_labels[DEPTH_PLUS1][v]=col;				//labels start at 1	
			}
			if((--pc)==0)
						return;
			m_sel.erase_block(nBB,g.get_neighbors(v));
		}				
	col++;
	}
}

template<class T>
void Clique<T>::expand(int maxac, typename T::bb_type& l_bb , lista_nodos_t& l_v){
////////////////////////
// recursive search algorithm

	int v;
//main loop
	while(l_v.index>=0){
			
		//Estrategias
		v=l_v.nodos[l_v.index--];

		//CUT by color (since [Konc & Janecic, 2007] this is of limited use (only for first branch))
		if( (m_lcol_labels[maxac][v]+maxac)<=maxno )
				return;
/////////////////////////////////
// CHILD NODE GENERATION
		
		//Node generation by masking
		AND(g.get_neighbors(v), l_bb, LISTA_BB(maxac));		//optimized when place second the bitset with higher population
		
		//Leaf node: updates incumbent if necessary
		if( LISTA_BB(maxac).is_empty()){
			if(maxac>=maxno){
				maxno=maxac+1;						//NEW GLOBAL OPTIMUM FOUND
				m_path[maxac]=v;
				cout<<"NEW lb:"<<maxno<<" root_v:"<<m_path[0]<<endl;
				copy(m_path, m_path+maxno, ostream_iterator<int>(cout, " ")); cout<<endl;
	
			}
		l_bb.erase_bit(v);
		continue;
		}
		//approx. coloring (generates child list of nodes in LISTA_L(maxac))
		paint(maxac);

		//cuts if there are no child nodes of v
		if(LISTA_L(maxac).index<0){
			l_bb.erase_bit(v);
			continue;
		}
				
///////////////////////////////////////////////////////
// CANDIDATE EXPANSION

		//sets path
		m_path[maxac]=v;
				
		//Generacion de nuevos nodos
		expand(maxac+1,LISTA_BB(maxac),LISTA_L(maxac));  

		//////////////////////////////////////////////
		// BACKTRACK (does not delete v fro path since it will be overwritten in the same level of search tree)
		l_bb.erase_bit(v); 
	}// next node

return;
}

template<class T>
void Clique<T>::expand_without_coloring(int maxac, typename T::bb_type& l_bb){
////////////////////////
// recursive search algorithm

	int v;
//main loop
	l_bb.init_scan(bbo:: DESTRUCTIVE_REVERSE);
	while(true){
			
		//Estrategias
		v=l_bb.previous_bit_del();
		if(v==EMPTY_ELEM) break;

/////////////////////////////////
// CHILD NODE GENERATION
		
		//Node generation by masking
		AND(g.get_neighbors(v),l_bb,LISTA_BB(maxac));
		
		//Leaf node: updates incumbent if necessary
		if( LISTA_BB(maxac).is_empty()){
			if(maxac>=maxno){
				maxno=maxac+1;						//NEW GLOBAL OPTIMUM FOUND
				m_path[maxac]=v;
				cout<<"NEW lb:"<<maxno<<" root_v:"<<m_path[0]<<endl;
				
			}
		continue;
		}			

///////////////////////////////////////////////////////
// CANDIDATE EXPANSION

		//sets path
		m_path[maxac]=v;
				
		//Generacion de nuevos nodos
		expand_without_coloring(maxac+1,LISTA_BB(maxac));  

		//////////////////////////////////////////////
		// BACKTRACK (does not delete v fron path since it will be overwritten in the same level of search tree)
		
	}// next node

return;
}


template<class T>
inline
int Clique<T>::set_up(){
	PrecisionTimer pt;
	
	KCore<T> kc(g);
	pt.wall_tic();
	cout<<"init kcore analysis----------------"<<endl;
    kc.kcore();
	cout<<"kcore ub:"<<kc.get_kcore_number()+1<<endl;
	cout<<"[t:"<<pt.wall_toc()<<"]"<<endl;

	//Calculo de lb for middle size graphs (remove for large sparse graphs)
	cout<<"init search for initial clique----------------"<<endl;
	pt.wall_tic();
	if(maxno>kc.get_kcore_number()){			//allows for an external initial value
		cout<<"[w:"<<maxno<<"]"<<" TRIVIALLY SOLVED DURING PRECOMPUTATION"<<endl;
		return maxno;
	}
	vector<int> v=kc.find_heur_clique();		 
	cout<<"[t:"<<pt.wall_toc()<<"]"<<endl;
	if(maxno<v.size())							//updates current best solution (allows initial value from other sources)
			maxno=v.size();
	if(maxno>kc.get_kcore_number()){
		cout<<"[w:"<<maxno<<"]"<<" TRIVIALLY SOLVED DURING PRECOMPUTATION"<<endl;
		return maxno;
	}
	cout<<"[lb:"<<maxno<<"]"<<endl;
		
	//Initial ordering for middle size graphs
	cout<<"init degeneracy reordering (macro)----------------"<<endl;
	InitOrder<T> o(g);	
	o.reorder(o.create_new_order(MIN_WIDTH_MIN_TIE_STATIC));
	
	//Initial order for large sparse graphs (MWS-kcore based)
	//cout<<"init degeneracy reordering (kcore-based)----------------"<<endl;
	//InitOrder<ugraph> o(g);	
	//const vint& kco=kc.get_kcore_ordering();
	////new order table
	//vint old2new(kco.size());
	//int l=0;
	//for(vint::const_reverse_iterator it=kco.rbegin(); it!=kco.rend(); ++it){
	//	old2new[*it]=l++;
	//}
	//cout<<"degeneracy reordering: init reordering (not in place) ---------------"<<endl;
	//o.reorder(old2new);					
	
	//Init search allocation
	cout<<"init search allocation----------------"<<endl;
	set_malloc(kc.get_kcore_number()+1);
	init_search_basics();
	init_color_labels();
	init_aux_data();

	//Color inicial
	cout<<"init initial coloring----------------"<<endl;
	InitColor<T> cinit(g);
	vint col;
	cinit.simpleDegreeColoring(col);
	copy(col.begin(), col.end(), m_lcol_labels[0]);


return 0;
}

template<class T>
void Clique<T>::run(bool info){
	
	//algorithm
	PrecisionTimer pt;
	double secs;
		
	if(info)
		pt.wall_tic();
	expand(0, m_bbroot, m_lroot);
	if(info)
		secs=pt.wall_toc();
		
	if(info)
		cout<<"[w:"<<maxno<<","<<secs<<"s]"<<endl;
}

template<class T>
void Clique<T>::run_without_coloring(bool info){
	
	//algorithm
	PrecisionTimer pt;
	double secs;

	if(info)
		pt.wall_tic();
	expand_without_coloring(0, m_bbroot);
	if(info)
		secs=pt.wall_toc();


	if(info)
		cout<<"[w:"<<maxno<<","<<secs<<"s]"<<endl;
}


template<class T>
void Clique<T>::run_unrolled(bool info){
////////////////
// runs search unrolling first level
	
	//algorithm
	PrecisionTimer pt;
	double secs;
	
	if(info)
		pt.wall_tic();
	initial_expand();
	if(info)
		secs=pt.wall_toc();


	if(info)
		cout<<"[w:"<<maxno<<","<<secs<<"s]"<<endl;
}

template<class T>
void Clique<T>::run_unrolled_without_coloring(bool info){
////////////////
// runs search unrolling first level
	
	//algorithm
	PrecisionTimer pt;
	double secs;

	if(info)
		pt.wall_tic();
	initial_expand_without_coloring(0, m_bbroot);
	if(info)
		secs=pt.wall_toc();

	if(info)
		cout<<"[w:"<<maxno<<","<<secs<<"s]"<<endl;
}

template<class T>
void Clique<T>::initial_expand(){
////////////////////
// unrolling of first level

	int v=EMPTY_ELEM;
	KCore<T> kc(g);
	InitColor<T> cinit(g);
	
	//Loop over neighbor set subproblems
	for(int v=m_size-1; v>=0; v--){
		LISTA_BB(0).init_bit(v,g.get_neighbors(v));
				
		//CUT related to size
		//if(LISTA_BB(0).popcn64()<=maxno){
		if(LISTA_BB(0).popcn64()<maxno){
	//		cout<<"PODA SIZE SUBPROBLEMA"<<endl;
			continue;
		}

		//COLOR CUT of this subproblem
		//if(cinit.greedyIndependentSetColoring(LISTA_BB(0))<=maxno){
		if(cinit.greedyIndependentSetColoring(LISTA_BB(0))<maxno){
		//	cout<<"PODA COLOR SUBPROBLEMA:"<<col<<endl;
			continue;
		}
				
		//kcore computation
		LISTA_BB(0).set_bit(v);								//add bit for kcore analysis only		
		kc.set_subgraph(&LISTA_BB(0));
		kc.kcore(); 

		//kcore graph number cut
		if(kc.get_kcore_number()<maxno){
			//	cout<<"PODA KCORE GRAPH:"<<col<<endl;
			continue;
		}
		
		//KCore cut
		LISTA_L(0).index=-1;
		const vint& kcn=kc.get_kcore_numbers();
		const vint& kcv=kc.get_kcore_ordering();
		for(int i=kcv.size()-1; i>=0; i--){
			if(kcn[kcv[i]]<maxno){
				//KCore cut for the subproblem
				for(int j=i; j>=0; j--){
					LISTA_BB(0).erase_bit(kcv[j]);		//O(logn) operation
				}
				break;
			}else{
				//add to candidate list for expansion
				if(kcv[i]!=v){
					LISTA_L(0).nodos[++LISTA_L(0).index]=kcv[i];
					m_lcol_labels[1][kcv[i]]=kcn[kcv[i]]+1;		
				}
			}
		}
	
	
//Expansion as in BBMC in minimum width order
		
		LISTA_BB(0).erase_bit(v);
		m_path[0]=v;
		expand(1,LISTA_BB(0),LISTA_L(0));		//Note: LISTA_L should not be empty: it would have been detected in KCORE-GRAPH CUT
	
// BACKTRACK  from v: vertex already deleted at the beginning of the iterations
	}

}

template<class T>
void Clique<T>::initial_expand_without_coloring(int maxac, typename T::bb_type& l_bb){
////////////////////
// unrolling of first level

	int v=EMPTY_ELEM;
	KCore<T> kc(g);
	InitColor<T> cinit(g);
	l_bb.init_scan(bbo::DESTRUCTIVE_REVERSE);
	while(true){
		v=l_bb.previous_bit_del();
		if(v==EMPTY_ELEM) break;
				
/////////////////////////////////
// NEW SUBPROBLEM

		//Node generation by masking
		AND(g.get_neighbors(v),l_bb, LISTA_BB(maxac));
		LISTA_BB(maxac).set_bit(v);								

		//CUT related to size
		if(LISTA_BB(maxac).popcn64()<=maxno){
	//		cout<<"PODA SIZE SUBPROBLEMA"<<endl;
			continue;
		}

		//COLOR CUT of this subproblem
		if( cinit.greedyIndependentSetColoring(LISTA_BB(maxac))<=maxno ){
	//		cout<<"PODA COLOR SUBPROBLEMA"<<endl;
			continue;
		}
				
		//kcore computation
		kc.set_subgraph(&LISTA_BB(maxac));
		kc.kcore(); 
		
		//KCore cut
		LISTA_L(maxac).index=-1;
		const vint& kcn=kc.get_kcore_numbers();
		const vint& kcv=kc.get_kcore_ordering();
		for(int i=kcv.size()-1; i>=0; i--){
			if(kcn[kcv[i]]<maxno){
				//KCore cut for the subproblem
				for(int j=i; j>=0; j--){
					LISTA_BB(maxac).erase_bit(kcv[j]);		//O(logn) operation
				}
				break;
			} 
		}
	
//Expansion as in BBMC in minimum width order
		
		LISTA_BB(maxac).erase_bit(v);
		m_path[0]=v;
		expand_without_coloring(maxac+1,LISTA_BB(maxac));
	
// BACKTRACK  from v: vertex already deleted at the beginning of the iterations
	}
}

template<class T>
void Clique<T>::print_filter (ostream& o){
	for(map<int, int>::iterator it= m_filter.begin(); it!=m_filter.end(); ++it){
		o<<"["<<it->first<<","<<it->second<<"]"<<" "; 
	}
	o<<endl;
}



#endif