//init_order.h: header for InitOrder implementation, a wrapper to order bit encoded undirected graphs by degree criteria
//date: 17/09/14
//authors: pablo san segundo, alvaro lopez

#ifndef __INIT_ORDER_H__
#define __INIT_ORDER_H__

#include "clique_types.h"
#include "pablodev/graph/graph.h"


//struct for vertex with information of vertex
struct deg_t{
	deg_t():index(EMPTY_ELEM), deg(0), deg_of_n(0){}
	int index;
	int deg;
	int deg_of_n;
};



struct degreeLess: 
	public binary_function<deg_t, deg_t, bool>{
	bool operator() (deg_t i,deg_t j) { 
		return (i.deg<j.deg);
	} 
};

struct degreeWithTieBreakLess: 
	public binary_function<deg_t, deg_t, bool>{
	bool operator() (deg_t i,deg_t j) const {
		if(i.deg<j.deg) return true;
		else if(i.deg==j.deg){
			if(i.deg_of_n<j.deg_of_n) return true;
		}
		return false;
	} 
};

typedef std::vector<int> vint;

///////////////////////////
//
// InitOrder class
// (only for ugraph and sparse_ugraph)
//
////////////////////////////

template <class T=ugraph>
class InitOrder{
public:
	static void print				(const vint& order, bool revert=false, ostream& o=std::cout);			
	InitOrder(T& gout):g(gout){}
	int reorder						(const vint& new_order, ostream* o = NULL);
	int reorder_in_place			(const vint& new_order, ostream* o=NULL);
	vint create_new_order			(init_order_t, place_t=PLACE_LF);										
	
	int deg_of_neighbors			(int v);								//computes support(sum of degree of neighbors)
////////////////
// data members
private:
	typedef std::vector<deg_t>				vdeg;							//vector of vertexs
	typedef std::vector<deg_t>::iterator	vdeg_it;						//iterator for vector of vertexs
	
	T& g;																	//T restricted to ugraph and sparse_ugraph
};

template<class T>
int InitOrder<T>::deg_of_neighbors(int v){
/////////////////////////
//Sum of degrees of neighbors to v in the current graph considered

	int ndeg=0,vadj=EMPTY_ELEM;
	if(g.get_neighbors(v).init_scan(BBObject::NON_DESTRUCTIVE)!=EMPTY_ELEM){
		while(true){
			vadj=((g.get_neighbors(v)).next_bit());
			if(vadj==EMPTY_ELEM) break;
			ndeg+=g.degree(vadj);
		}
	}
return ndeg;
}

template<class T>
void InitOrder<T>::print(const std::vector<int>& new_order, bool revert, ostream& o){
	o<<"new order: ";
	if(revert){
		copy(new_order.rbegin(), new_order.rend(), ostream_iterator<int>(o, " "));
	}else{
		copy(new_order.begin(), new_order.end(), ostream_iterator<int>(o, " "));
	}
o<<endl;
}

template<class T>
inline
int InitOrder<T>::reorder(const vint& new_order, ostream* o){
/////////////////////
// reordering in place
// new order logs to "o"
//
// REMARKS: 
// 1-Experimental (uses auxiliary graph: should be done truly in place)
// 2-only for undirected graphs

	//control
	if(new_order.size()!=g.number_of_vertices()){
		cerr<<"inconsistent ordering: cannot reorder graph"<<endl;
		return -1;
	}

	int size=g.number_of_vertices();
	T gn(size);
	gn.set_name(g.get_name());
	
	//only for undirected graphs
	for(int i=0; i<size-1; i++){
		for(int j=i+1; j<size; j++){
			if(g.is_edge(i,j)){								//in O(log) for sparse graphs, should be specialized for that case
				//switch edges according to new numbering
				gn.add_edge(new_order[i], new_order[j]);
			}
		}
	}
	g=gn;
			
	//new order to stream if available
	if(o!=NULL)
		copy(new_order.begin(), new_order.end(), ostream_iterator<int>(*o, " "));
	
return 0;		
}

template<>
inline
int InitOrder<sparse_ugraph>::reorder(const vint& new_order, ostream* o){
/////////////////////
// specialization for sparse graphs
//
// REMARKS: 
// 1-Experimental (uses auxiliary graph: should be done truly in place)
// 2-only for undirected graphs

	//control
	if(new_order.size()!=g.number_of_vertices()){
		cerr<<"inconsistent ordering: cannot reorder sparse undirected graph"<<endl;
		return -1;
	}

	int size=g.number_of_vertices();
	sparse_ugraph gn(size);
	gn.set_name(g.get_name());
	
	//only for undirected graphs
	int j=EMPTY_ELEM;
	for(int i=0; i<size-1; i++){
		sparse_bitarray neigh=g.get_neighbors(i);
		if(neigh.init_scan_from(i, bbo::NON_DESTRUCTIVE)!=EMPTY_ELEM){
			while(true){
				j=neigh.next_bit();
				if(j==EMPTY_ELEM)
						break;
				gn.add_edge(new_order[i], new_order[j]);
			}
		}
	}
	g=gn;
			
	//new order to stream if available
	if(o!=NULL)
		copy(new_order.begin(), new_order.end(), ostream_iterator<int>(*o, " "));
	
return 0;		
}

template<>
inline
int InitOrder<sparse_ugraph>::reorder_in_place(const vint& new_order, ostream* o){
/////////////////////
// Reorders graph in place (only for large sparse UNDIRECTED graphs)
// date: 22/12/14
// author: Alvaro Lopez
//
// COMMENTS: Opimmize for space requirements

    //control
	int N=g.number_of_vertices();
	if(new_order.size()!=N){
        cerr<<"inconsistent ordering: cannot reorder sparse undirected graph"<<endl;
        return -1;
    }
	
   	
	//Deletes lower triangle of adjacency matrix
	cout<<"deleting low triangle--------------------"<<endl;
    for(int i=0; i<N; i++){
        g.get_neighbors(i).clear_bit(0,i);
		g.get_neighbors(i).shrink_to_fit();
	}
		
	cout<<"new order upper to lower triangle--------------------"<<endl;		
	sparse_bitarray neigh;
    int j=EMPTY_ELEM;
    for(int i=0; i<N; i++){
        neigh=g.get_neighbors(i);
		//reorders using upper triangle information
        if(neigh.init_scan_from(i, bbo::NON_DESTRUCTIVE)!=EMPTY_ELEM){
		    while(true){
                j=neigh.next_bit();
                if(j==EMPTY_ELEM)
                        break;
			
				//writes new edge in lower triangle
                if(new_order[i]>new_order[j]){
					g.get_neighbors(new_order[i]).set_bit(new_order[j]);
				} else{
                    g.get_neighbors(new_order[j]).set_bit(new_order[i]);
				}
            }
        }
		//Deletes each neighborhood once read
		g.get_neighbors(i).clear_bit(i, N-1);
		g.get_neighbors(i).shrink_to_fit();
    }


    //Makes the graph bidirected: copies in54.-51kl,.5formation from lower to upper triangle
	cout<<"making graph bidirected--------------------"<<endl;	
    for(int i=0; i<N; i++){
        neigh=g.get_neighbors(i);
        if(neigh.init_scan(bbo::NON_DESTRUCTIVE)!=EMPTY_ELEM){ 
            while(true){
                j=neigh.next_bit();
                if((j==EMPTY_ELEM) || (j>i))
                        break;
				g.get_neighbors(j).set_bit(i);
	        }
        }
    }
	   
    //new order to stream if available
    if(o!=NULL)
        copy(new_order.begin(), new_order.end(), ostream_iterator<int>(*o, " "));
  return 0;        
}

template<class T>
inline
vint InitOrder<T>::create_new_order (init_order_t  alg, place_t place){
/////////////////////////////
// Ordena por estrategias recorriendo los vrtices en orden estricto creciente
// Si LF (last to first) es TRUE los coloca empezando por el final, si FALSE los coloca empezando por el principio
//
// As usual new order[OLD_INDEX]=NEW_INDEX
// RETURNS: Empty vertex if ERROR, else new ordering
//
// REMARKS
// 1.Had to make tie-breaks more efficient (28/8/14)
// 2.There was a lot to do! Basically degrees with respect to the vertex removed and support can be recomputed over the updated degrees.
			
	vint new_order(g.number_of_vertices());
	vdeg degs;					
	int k;
	(place==PLACE_LF)? k=g.number_of_vertices()-1 : k=0;
			
	//computes degree of vertices
	for(int i=0; i<g.number_of_vertices(); i++){
		deg_t vt;
		vt.index=i;
		vt.deg=g.degree(i);
		vt.deg_of_n=deg_of_neighbors(vt.index);
		degs.push_back(vt);		
	}
		
	 if(alg==MIN_WIDTH){
		BitBoardN bbn(g.number_of_vertices()); bbn.set_bit(0,g.number_of_vertices()-1);
		while(!degs.empty()){
			vdeg_it it1=min_element(degs.begin(), degs.end(), degreeLess());
			new_order[it1->index]=k;
			(place==PLACE_LF)? k-- : k++;
			bbn.erase_bit(it1->index);
			degs.erase(it1);

			//recompute degrees
			for(int i=0; i<degs.size(); i++){
				degs[i].deg=g.degree(degs[i].index, bbn);
			}
		}
	}else if(alg==MIN_WIDTH_MIN_TIE_STATIC){
		BitBoardN bbn(g.number_of_vertices()); bbn.set_bit(0,g.number_of_vertices()-1);
		while(!degs.empty()){
			vdeg_it it_sel=min_element(degs.begin(), degs.end(), degreeWithTieBreakLess());
			int v_sel=it_sel->index;
			new_order[v_sel]=k;
			(place==PLACE_LF)? k-- : k++;
			bbn.erase_bit(v_sel);
			degs.erase(it_sel);
							
			//recompute degrees 
			for(int i=0; i<degs.size(); i++){
				degs[i].deg=g.degree(degs[i].index,bbn);
			}
		}

	}else if(alg==NONE){
		for(int i=0; i<new_order.size(); i++){
			new_order[i]=k;
			(place==PLACE_LF)? k-- : k++;
		}
		
	}else {			
		vint().swap(new_order);
	}

return new_order;
}


#endif