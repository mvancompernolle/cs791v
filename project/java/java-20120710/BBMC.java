import java.util.*;

public class BBMC extends MCQ {

    BitSet[] N;    // neighbourhood
    BitSet[] invN; // inverse neighbourhood
    Vertex[] V;    // mapping bits to vertices

    BBMC (int n,int[][]A,int[] degree,int style) {
	super(n,A,degree,style);
	N    = new BitSet[n];
	invN = new BitSet[n];
	V    = new Vertex[n];
    }
    
    void search(){
	cpuTime  = System.currentTimeMillis();
	nodes    = 0;
	BitSet C = new BitSet(n);
	BitSet P = new BitSet(n);

	for (int i=0;i<n;i++){	  
	    N[i]    = new BitSet(n);
	    invN[i] = new BitSet(n);
	    V[i]    = new Vertex(i,degree[i]);
	}

System.out.print("V: (index / degree)\n");
for(Vertex v: V){
	System.out.print(v.index + "-" + v.degree + " ");
}
System.out.print("\n");

	orderVertices();
	C.set(0,n,false);
	P.set(0,n,true);
/*
printBitset(C);
printBitset(P);
for(int i=0; i<n; i++)
	printBitset(invN[i]);
*/
	BBMaxClique(C,P);
    }
    
    void BBMaxClique(BitSet C,BitSet P){
//System.out.print("\n");
//printBitset(C);
//printBitset(P);

	if (timeLimit > 0 && System.currentTimeMillis() - cpuTime >= timeLimit) return;
	nodes++;
	int m = P.cardinality();

	int[] U = new int[m];
	int[] colour = new int[m];
	BBColour(P,U,colour);

//printArray(U, m);
//printArray(colour, m);

	for (int i=m-1;i>=0;i--){
	    if (colour[i] + C.cardinality() <= maxSize) return;
	    BitSet newP = (BitSet)P.clone();	    
	    int v = U[i];
	    C.set(v,true); newP.and(N[v]);
/*
System.out.print("N: \n");
printBitset(N[v]);
System.out.print("newP: \n");
printBitset(newP);
*/
	    if (newP.isEmpty() && C.cardinality() > maxSize) {
	    	//System.out.print("saved\n");
	    	saveSolution(C);
	    }
	    if (!newP.isEmpty()) {
	    	//System.out.print("called again\n");
	    	BBMaxClique(C,newP);
	    }
	    P.set(v,false); C.set(v,false);
	}
    }
    
    void BBColour(BitSet P,int[] U,int[] colour){
	BitSet copyP    = (BitSet)P.clone();
	int colourClass = 0;
	int i           = 0;
       	while (copyP.cardinality() != 0){
	    colourClass++;
	    BitSet Q = (BitSet)copyP.clone();
	    while (Q.cardinality() != 0){
		int v = Q.nextSetBit(0);
		copyP.set(v,false);
		Q.set(v,false);
		Q.and(invN[v]);
		U[i] = v;     
		colour[i++] = colourClass;
	    }
	}
    }

    void orderVertices(){
	for (int i=0;i<n;i++){
	    V[i] = new Vertex(i,degree[i]);
	    for (int j=0;j<n;j++) 
		if (A[i][j] == 1) V[i].nebDeg = V[i].nebDeg + degree[j];
	}
        if (style == 1) Arrays.sort(V); 
	if (style == 2) minWidthOrder(V);    
	if (style == 3) Arrays.sort(V,new MCRComparator());

System.out.print("V: (index / degree)\n");
for(Vertex v: V){
	System.out.print(v.index + "-" + v.degree + " ");
}
System.out.print("\n");

	for (int i=0;i<n;i++)
	    for (int j=0;j<n;j++){
		int u = V[i].index;
		int v = V[j].index;
		N[i].set(j,A[u][v] == 1);
		invN[i].set(j,A[u][v] == 0);
	    }
    }

    void saveSolution(BitSet C){
	Arrays.fill(solution,0);
	for (int i=0;i<C.size();i++) if (C.get(i)) solution[V[i].index] = 1;
	maxSize = C.cardinality();
    }

    void printBitset(BitSet b){
    	for(int i=0; i<n; i++){
    		int v = 0;
    		if(b.get(i))
    			v = 1;
    		System.out.print(v + " ");
    	}
    	System.out.print("\n");
    }

    void printArray(int[] b, int size){
    	for(int i=0; i<size; i++){
    		System.out.print(b[i] + " ");
    	}
    	System.out.print("\n");
    }
}
