import subprocess as sub
import os, sys

def main():

    L = []

    #L.append("BBMC1")
    L.append("BBMC2")
    #L.append("BBMC3")
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
    #G.append("../java/DIMACS_cliques/brock200_1.clq")
    #G.append("../java/DIMACS_cliques/brock400_4.clq")
    #G.append("../java/DIMACS_cliques/MANN_a27.clq")
    #G.append("../java/DIMACS_cliques/p_hat1000-1.clq")
    #G.append("../java/DIMACS_cliques/p_hat1500-1.clq")
    #G.append("../java/DIMACS_cliques/p_hat300-3.clq")
    #G.append("../java/DIMACS_cliques/p_hat500-2.clq")
    #G.append("../java/DIMACS_cliques/p_hat500-3.clq")
    #G.append("../java/DIMACS_cliques/p_hat700-2.clq")
    #G.append("../java/DIMACS_cliques/san1000.clq")
    #G.append("../java/DIMACS_cliques/san200_0.9_2.clq")
    #G.append("../java/DIMACS_cliques/san200_0.9_3.clq")
    #G.append("../java/DIMACS_cliques/san400_0.7_1.clq")
    #G.append("../java/DIMACS_cliques/san400_0.7_2.clq")
    #G.append("../java/DIMACS_cliques/san400_0.7_3.clq")
    #G.append("../java/DIMACS_cliques/sanr200_0.9.clq")
    #G.append("../java/DIMACS_cliques/sanr400_0.5.clq")
    #G.append("../java/DIMACS_cliques/sanr400_0.7.clq")

    #G.append("../java/DIMACS_cliques/san400_0.9_1.clq")
    #G.append("../java/DIMACS_cliques/p_hat1000-2.clq")
    #G.append("../java/DIMACS_cliques/p_hat500-3.clq")
    #G.append("../java/DIMACS_cliques/brock400_3.clq")
    #G.append("../java/DIMACS_cliques/brock400_2.clq")
    #G.append("../java/DIMACS_cliques/brock400_1.clq")
    #G.append("../java/DIMACS_cliques/brock800_4.clq")



    G.append("../200_3.txt")
    G.append("../200_4.txt")
    G.append("../200_5.txt")
    G.append("../200_6.txt")
    G.append("../200_7.txt")
    G.append("../200_8.txt")

    G.append("../300_3.txt")
    G.append("../300_4.txt")
    G.append("../300_5.txt")
    G.append("../300_6.txt")
    G.append("../300_7.txt")
    G.append("../300_8.txt")

    G.append("../400_3.txt")
    G.append("../400_4.txt")
    G.append("../400_5.txt")
    G.append("../400_6.txt")
    G.append("../400_7.txt")
    G.append("../400_8.txt")

    G.append("../500_3.txt")
    G.append("../500_4.txt")
    G.append("../500_5.txt")
    G.append("../500_6.txt")
    G.append("../500_7.txt")
    G.append("../500_8.txt")

    G.append("../600_3.txt")
    G.append("../600_4.txt")
    G.append("../600_5.txt")
    G.append("../600_6.txt")
    G.append("../600_7.txt")
    G.append("../600_8.txt")

    with open('log.txt', 'a') as fout:

        for graph in G:

            for runType in L:

                # spawn a process that runs the linter on the file
                p = sub.Popen("./project " + runType + " " + graph + " 3600", stdout=sub.PIPE, stderr=sub.PIPE, shell=True)

                # get the num errors from the linter and write it to the data file
                fout.write("NEW LOG ENTRY\n")
                out, err = p.communicate()
                fout.write("Output: " + out + "\nError: " + err + "\n\n")

if __name__ == '__main__':
    main()