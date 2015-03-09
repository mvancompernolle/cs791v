//init_color.h: header for InitColor implementation, a wrapper to color bit encoded
//				undirected graphs using greedy independent set heuristic
//
//date of creation: 23/09/14
//last update: 23/09/14
//author: pablo san segundo

#ifndef __INIT_COLOR_H__
#define __INIT_COLOR_H__

#include "pablodev/graph/graph.h"
#include <iterator>

#define MAX_NUM_COLORS 1000			
typedef vector<int> vint;

///////////////////////////
//
// InitOrder class
// (only for ugraph and sparse_ugraph)
//
////////////////////////////
template < class T >
class InitColor{
public:
	~InitColor(){clear_ColorMatrix();}
	InitColor(T& gout):g(gout),m_unsel(g.number_of_vertices()),  m_sel(g.number_of_vertices()), m_color_matrix(NULL)	{}
	int greedyIndependentSetColoring(vint& );
	int greedyIndependentSetColoring(typename T::bb_type &);			//subgraph coloring
	int greedyColoring(vint&);
	int simpleDegreeColoring(vint& );
	int greedyColorMatrixColoring(vint&);								

public:
	void init_ColorMatrix();											
	void clear_ColorMatrix();

private:	
	T& g;																//T restricted to ugraph and sparse_ugraph
	//auxiliary data structures
	typename T::bb_type m_unsel;
	typename T::bb_type m_sel;
	typename T::bb_type* m_color_matrix;								//used in greedy matrix coloring (12/14)	
};

template < class T >
void InitColor<T>::clear_ColorMatrix(){
	if(m_color_matrix){
		delete [] m_color_matrix;
		m_color_matrix=NULL;
	}
}

template < class T >
void InitColor<T>::init_ColorMatrix(){
	clear_ColorMatrix();
	m_color_matrix=new typename T::bb_type[MAX_NUM_COLORS];
	for(int i=0; i<MAX_NUM_COLORS; i++){
		m_color_matrix[i].init(g.number_of_vertices());
	}
}

template < class T >
inline
int InitColor<T>::greedyIndependentSetColoring(vint& color){
/////////////////////
// computes color classes iteratively

	int pc=g.number_of_vertices(), col=1, v=EMPTY_ELEM, from=EMPTY_ELEM;
	color.assign(pc, EMPTY_ELEM);		
	
  
	m_unsel.set_bit(0,pc-1);
	
	while(true){
		m_sel=m_unsel;
		m_sel.init_scan(bbo::DESTRUCTIVE);							
		while(true){
			if((v=m_sel.next_bit_del(from, m_unsel))==EMPTY_ELEM) 
				break;
			color[v]=col;

			if((--pc)==0)	
				return col;
			
			//computes next vertex of the current color class
			m_sel.erase_block(from, g.get_neighbors(v));
		}

	++col;
	}

return col;		//should not reach here
}

template<>
inline
int InitColor<ugraph>::greedyIndependentSetColoring(bitarray & bb){
/////////////////////
// computes color classes iteratively : specialization for the non sparse case
//
// REMARKS: bb has to have the number of bitblocks to hold all vertices of the original graph 
//          independent of the size of the subgraph

	int pc=bb.popcn64(), col=1, v=EMPTY_ELEM, from=EMPTY_ELEM;

	//empty check
	if(pc==0)
			return 0;

   
	m_unsel=bb;

	while(true){
		m_sel=m_unsel;
		m_sel.init_scan(bbo::DESTRUCTIVE);
		while(true){
			if( (v=m_sel.next_bit_del(from, m_unsel)) ==EMPTY_ELEM ) 
				  break;

			if((--pc)==0)	
				 return col;
			
			//computes next vertex of the current color class
			m_sel.erase_block(from, g.get_neighbors(v));
		}
	col++;
	}

return col;								
}

template<>
inline
int InitColor<sparse_ugraph>::greedyIndependentSetColoring(sparse_bitarray & bb){
///////////////////
// computes color classes iteratively:	specialization for the sparse case
	int pc=bb.popcn64(), col=1, v=EMPTY_ELEM, from=EMPTY_ELEM;

	//empty check
	if(pc==0)
			return 0;
		
						//(1)
	m_unsel=bb;
		
    sparse_bitarray::velem_it from_it;
	while(true){
		m_sel=m_unsel;
		m_sel.init_scan(bbo::DESTRUCTIVE);			//empty condition not tested since it can never hold
											
		from_it=m_unsel.begin();
		while(true){
			if((v=m_sel.next_bit_del(from))==EMPTY_ELEM)						
				break;
		
			if((--pc)==0)	
				return col;
			
			//optimized erasing of bit at each iteration
			from_it=m_unsel.erase_bit(v, from_it);			 //in O(log N): optimization so that the lookup is restricted from the last search onwards

			//computes next vertex of the current color class
			m_sel.erase_block(from, g.get_neighbors(v));
		}
	++col;
	}
return col;			//should not reach here					
}
	

template<>
inline
int InitColor<sparse_ugraph>::greedyIndependentSetColoring(vint & color){
///////////////////
// specialization for the sparse case
	int pc=g.number_of_vertices(), col=1, v=EMPTY_ELEM, from=EMPTY_ELEM;

	//empty check
	if(pc==0)
			return 0;

	//init assignmemt
	color.assign(pc, EMPTY_ELEM);

	m_unsel.init_bit(0,pc-1);
	
    sparse_ugraph::bb_type::velem_it from_it;
	while(true){
		m_sel=m_unsel;
		if(m_sel.init_scan(bbo::DESTRUCTIVE)==EMPTY_ELEM) return col;			//this test should be unncessary
		from_it=m_unsel.begin();
		
		while(true){
			if((v=m_sel.next_bit_del(from))==EMPTY_ELEM)						
				break;
			color[v]=col;

			if((--pc)==0)	
				return col;
			
			//optimized erasing of bit at each iteration
			from_it=m_unsel.erase_bit(v, from_it);							 //in O(log N): optimization so that the lookup is restricted from the last search onwards

			//computes next vertex of the current color class
			m_sel.erase_block(from, g.get_neighbors(v));
		}
	col++;
	}
return col;			//should not reach here
}


template<class T>
inline
int InitColor<T>::greedyColoring(vint& color){
///////////////////////
// classical sequential coloring, labelling vertices as presented to the algorithm
//
// REMAKS: this is just experimental, to test the BITSCAN / GRAPH library

	const int NV=g.number_of_vertices();
	color.assign(NV, EMPTY_ELEM);
	
	int cmax=1, v=EMPTY_ELEM;

	typename T::bb_type color_classes[MAX_NUM_COLORS];
	typename T::bb_type uncol(NV);
	bool good_col;

	//init data structures
	color_classes[cmax].init(NV);	
	uncol.set_bit(0,NV-1);				

	uncol.init_scan(bbo::DESTRUCTIVE);
	while(true){
		v=uncol.next_bit_del();
		if(v==EMPTY_ELEM) break;
		for(int col=1; col<=cmax; col++){
			good_col=g.get_neighbors(v).is_disjoint(color_classes[col]);
			if (good_col){
				//existing color used
				color_classes[col].set_bit(v);										  
				color[v]=col; 
				cout<<col<<":"<<color_classes[col]<<endl;
				break;
			}
				
		}

		//new color	
		if (!good_col){
				++cmax;
				color_classes[cmax].init(NV);
				color_classes[cmax].set_bit(v);										
				color[v]=cmax;
		}
	}

return cmax;	
}

template<>
inline
int InitColor<sparse_ugraph>::greedyColoring(vint& color ){
///////////////////////
// Specialization for sparse undirected graphs (only affects the set_bit function, changed for init_bit)

	const int NV=g.number_of_vertices();
	color.assign(NV, EMPTY_ELEM);
	int cmax=1, v=EMPTY_ELEM;
	bool good_col=true;

	sparse_bitarray color_classes[MAX_NUM_COLORS];
	sparse_bitarray uncol(NV);
	

	//init data structures
	color_classes[cmax].init(NV);	
	uncol.init_bit(0,NV-1);				 
	
	uncol.init_scan(bbo::DESTRUCTIVE);
	while(true){
		v=uncol.next_bit_del();
		if(v==EMPTY_ELEM) break;
		for(int col=1; col<=cmax; col++){
			good_col=g.get_neighbors(v).is_disjoint(color_classes[col]);
			if (good_col){
				//existing color used
				color_classes[col].set_bit(v);										  
				color[v]=col; 
				cout<<col<<":"<<color_classes[col]<<endl;
				break;
			}
		}

		//new color	
		if (!good_col){
				++cmax;
				color_classes[cmax].init(NV);
				color_classes[cmax].init_bit(v);									
				color[v]=cmax;
		}
	}

return cmax;	
}

template<class T>
inline
int InitColor<T>::simpleDegreeColoring(vint& color){
////////////////////
// trivial coloring used in root node in BBMC
//
// Experimental: check color size

	int NV= g.number_of_vertices();
	color.assign(NV, EMPTY_ELEM);
	int gdeg=g.max_degree_of_graph();

	for(int i=0; i<gdeg;i++)
				 color[i]=i+1;			
	for(int i=gdeg; i<NV; i++)
				 color[i]=gdeg+1;  
		
return gdeg+1;				
}

template<class T>
inline
int InitColor<T>::greedyColorMatrixColoring(vint& color){
////////////////////
// New implementation of Sequential Coloring using a color matrix to optimize speed
// in a bit-parallel framework
//
// OBSERVATIONS: Requires previous call to init_ColorMatrix()
//
// Original idea: Larisa Komosko (LATNA), described in the paper 
// [A fast greedy sequential heuristic for the vertex colouring problem based on bitwise operations, 2014]
//	(currently under review)

// date:21/12/14

    int NV=g.number_of_vertices(), cmax=1, v=EMPTY_ELEM, from=EMPTY_ELEM;
    color.assign(NV, EMPTY_ELEM);

	//initializes first bit of first color of color_matrix to 0 so that the first vertex
	//is assigned color 1
	m_color_matrix[cmax].erase_bit(0);
	
	//main nested loop: outer->vertices, inner->possible color assignments
	for(int v=0; v<NV; v++){
		for(int c=1; c<=cmax; c++){
			if(!m_color_matrix[c].is_bit(v)){		//checks if color is available
				color[v]=c;
				m_color_matrix[c].set_block(WDIV(v),g.get_neighbors(v));					//OR operation
				goto next_vertex;
			}
		}

		//new color
		cmax++;
		color[v]=cmax;
		m_color_matrix[cmax].copy_from_block(WDIV(v),g.get_neighbors(v));					//overwrite operation	
		
next_vertex:
			;
	}
  
   return cmax;


   //Alternative implementation: Alvaro Lopez
   //for(int v=0; v<NV; v++){
   //for(int c=1; c<MAX_NUM_COLORS; c++){
   //	if(!m_color_matrix[c].is_bit(v)){
   //		color[v]=c;									//1-based
   //		//selective update (useful if the same color matrix is used for different colorings)
   //		if(c>col_size){						
   //			col_size=c;
   //			m_color_matrix[c].copy_from_block(WDIV(v),g.get_neighbors(v));		
   //			//colorMatrix[c].set_block(WDIV(v),g.get_neighbors(v));
   //			//colorMatrix[c]=g.get_neighbors(v);			// falta una funcion void init (int first block, const BitBoardN& );
   //		}
   //		else
   //			m_color_matrix[c].set_block(WDIV(v),g.get_neighbors(v));
   //		break;
   //	}
   //}
   //}

}



#endif