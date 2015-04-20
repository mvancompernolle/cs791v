import subprocess as sub
import os, sys

def main():

    N = []
    N.append("1");

    L = []
    L.append("BBMC2")
    L.append("BBMC1")
    L.append("BBMC3")
    #L.append("MCSA1")
    #L.append("MCSA2")
    #L.append("MCSA3")
    #L.append("MCSB1")
    #L.append("MCSB2")
    #L.append("MCSB3")
    #L.append("MCQ1")
    #L.append("MCQ2")
    #L.append("MCQ3")

    G = []

    G.append("../200_3.txt")
    G.append("../200_4.txt")
    G.append("../200_5.txt")
    G.append("../200_6.txt")
    G.append("../200_7.txt")
    G.append("../200_8.txt")
    G.append("../200_9.txt")

    G.append("../300_5.txt")
    G.append("../400_5.txt")
    G.append("../500_5.txt")
    G.append("../600_5.txt")
    G.append("../700_5.txt")
    G.append("../800_5.txt")
    G.append("../900_5.txt")
    G.append("../1000_5.txt")

    G.append("../java/DIMACS_cliques/brock200_1.clq")
    G.append("../java/DIMACS_cliques/brock400_4.clq")
    G.append("../java/DIMACS_cliques/MANN_a27.clq")
    G.append("../java/DIMACS_cliques/p_hat1000-1.clq")
    G.append("../java/DIMACS_cliques/p_hat1500-1.clq")
    G.append("../java/DIMACS_cliques/p_hat300-3.clq")
    G.append("../java/DIMACS_cliques/p_hat500-2.clq")
    G.append("../java/DIMACS_cliques/p_hat500-3.clq")
    G.append("../java/DIMACS_cliques/p_hat700-2.clq")
    G.append("../java/DIMACS_cliques/san1000.clq")
    G.append("../java/DIMACS_cliques/san200_0.9_2.clq")
    G.append("../java/DIMACS_cliques/san200_0.9_3.clq")
    G.append("../java/DIMACS_cliques/san400_0.7_1.clq")
    G.append("../java/DIMACS_cliques/san400_0.7_2.clq")
    G.append("../java/DIMACS_cliques/san400_0.7_3.clq")
    G.append("../java/DIMACS_cliques/sanr200_0.9.clq")
    G.append("../java/DIMACS_cliques/sanr400_0.5.clq")
    G.append("../java/DIMACS_cliques/sanr400_0.7.clq")
    G.append("../java/DIMACS_cliques/san400_0.9_1.clq")
    G.append("../java/DIMACS_cliques/p_hat1000-2.clq")
    G.append("../java/DIMACS_cliques/p_hat500-3.clq")
    G.append("../java/DIMACS_cliques/brock400_3.clq")
    G.append("../java/DIMACS_cliques/brock400_2.clq")
    G.append("../java/DIMACS_cliques/brock400_1.clq")
    G.append("../java/DIMACS_cliques/brock800_4.clq")

    with open('log.txt', 'a') as fout:

        for num in N:

            for runType in L:

                for graph in G:

                    # spawn a process that runs the linter on the file
                    p = sub.Popen("./project " + runType + " " + graph + " " + num + " 3600", stdout=sub.PIPE, stderr=sub.PIPE, shell=True)

                    # get the num errors from the linter and write it to the data file
                    fout.write("NEW LOG ENTRY\n")
                    out, err = p.communicate()
                    fout.write("Output: " + out + "\nError: " + err + "\n\n")

                    print("Graph: " + graph + " Run Type: " + runType + " Num: " + num + " done.")

if __name__ == '__main__':
    main()