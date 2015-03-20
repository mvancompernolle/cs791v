import subprocess as sub
import os, sys

def main():

    L = []
    L.append("BBMC1 ../java/DIMACS_cliques/brock200_1.clq")
    L.append("BBMC2 ../java/DIMACS_cliques/brock200_1.clq")
    L.append("BBMC3 ../java/DIMACS_cliques/brock200_1.clq")

    with open('log.txt', 'a') as fout:

        for runType in L:

            # spawn a process that runs the linter on the file
            p = sub.Popen("./project " + runType, stdout=sub.PIPE, stderr=sub.PIPE, shell=True)

            # get the num errors from the linter and write it to the data file
            fout.write("NEW LOG ENTRY\n")
            out, err = p.communicate()
            fout.write("Output: " + out + "\nError: " + err + "\n\n")

if __name__ == '__main__':
    main()