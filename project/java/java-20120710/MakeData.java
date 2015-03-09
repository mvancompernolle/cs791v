
public class MakeData {

    public static void makeJob(int n,double p,int m) {
	int p100   = (int)(p*100.0);
	String s = "-";
	if (p100 < 10) s = "-0";
	for (int i=0;i<m;i++){
	    System.out.format("java RandomGraph %d %.2f ",n,p);
	    if (i>9)
		System.out.println(" > ../randomData/"+ n + s +  p100 +"-"+ i +".txt");
	    else
		System.out.println(" > ../randomData/"+ n + s +  p100 +"-0"+ i +".txt");
	}
    }

    public static void main(String[] args) {
	int n        = Integer.parseInt(args[0]);   // vertices in a graph
	double pLo   = Double.parseDouble(args[1]); // starting probability
	double pHi   = Double.parseDouble(args[2]); // ending probability
	double pInc  = Double.parseDouble(args[3]); // p increment
	int m        = Integer.parseInt(args[4]);     // number of graphs at each setting

	double p = pLo;       
	for (int i=0;p<=pHi+(pInc/2.0);i++){
	    makeJob(n,p,m);
	    p = p + pInc;
	}
    }
}
