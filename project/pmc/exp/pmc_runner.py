'''
@author: Ryan Rossi

This script is used for evaluating the parallel maximum clique framework.

@python2.6 pmc_runner.py <tscc, output, dimacs, all> <exact, heur, all> <pmc, all>
Creates sh files for each graph, and finds the max clique of each graph, one at a time.


The first parameter is a comma-seperated list of the exact folders in the data directory 
containing graphs of that type. For instance, if 'social_networks, dimacs, tscc' is given 
as input, then the maximum clique for all the graphs in each directory are computed.

@python2.6 pmc.py 
1. graph collection: social, tscc, DIMACS 
2. algorithm: 0,1,2,...
3. number of threads: 1,2,4,8,16,32,64,128,...
4. name of folder to be created to store the results in benchmark directory
5. use 'stats' to compute a bunch of other graph stats/bounds
6. NUMA settings, use 'membind' or 'interleave', leave blank otherwise
7. include only graphs matching a given substring, for example: 'soc-'

Example:
python pmc_runner.py 'social' '0,2' '1,2' results 0 0 'socfb-'

After running this script, generate the tables/data for plots, using:
python pmc_perf_table.py results
'''

__author__      = "Ryan A. Rossi"
__copyright__   = "Copyright 2012-2013, Ryan A. Rossi"
__email__ 	= "rrossi@purdue.edu"
__version__ 	= "1.0"

import sys
import os
from sets import Set
import errno
import pmc_utils
from optparse import OptionParser

def ensure_path_exists(path):
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise


if __name__ == "__main__":
	
	graph_type = []
	if len(sys.argv) > 1:
		graph_types = sys.argv[1].split(',')
		print graph_types
	
	eval = []
	if len(sys.argv) > 2:
		eval = sys.argv[2].split(',')
		print eval
		
	threads = 1;
	if len(sys.argv) > 3:
		threads = sys.argv[3].split(',')
		print threads
	
	if len(sys.argv) > 4:
		results_folder = sys.argv[4]
		
	stats_flag = '';
	if len(sys.argv) > 5:
		stats_flag = sys.argv[5]
		
	numactl_flag = ''
	numactl_str = ''
	if len(sys.argv) > 6:
		numactl_flag = sys.argv[6]
	if "membind" in numactl_flag:
		numactl_flag = 'numactl --membind=all '
		numactl_str = '_membind'
	elif 'interleave' in numactl_flag:
		numactl_flag = 'numactl --interleave=all '
		numactl_str = '_interleave'
	else:
		numactl_flag = ''
		
		
	#setup path and dir to store results
	dir_str = 'tmp'
	sh_path = results_folder + '/' + dir_str + '/'
	ensure_path_exists(sh_path)
	
	#create sh file for each graph and alg evaluated
	fn_list = []
	sh_list = []
	prob = ''
	for gtype in graph_types:
		print gtype
		path="../data/" + gtype + '/'
		dirList=os.listdir(path)
		print dirList
		if len(sys.argv) > 7:
			filter_graphs = sys.argv[7]
			dirList = [i for i in dirList if filter_graphs in i]
			print dirList
	
		for fname in dirList:
			print fname
			if len(fname.split('.')[0]) > 0 and '.DS' not in fname and '.mtx' in fname or '.txt' in fname: # and '.edges' in fname and '.txt' in fname:
				fn_list.append(fname)
				for t in threads:
					for alg in eval:
						mcfinder = alg
						print alg
						print eval
						if int(mcfinder) > 9:
							prob = 'enum'
						else:
							prob = 'mc'
						
						shfile = ''
						shfile = gtype + '_alg' + alg + '_' + prob + '_t' + str(t) + numactl_str + '_' + fname.split('.')[0] + '.sh'
						print shfile
						fout = open(sh_path + shfile, "w+")
						fout.write('#!/bin/bash\n\n# nohup sh ./pmc.sh >& pmc_perf.txt  &\n#\n#\n\n\n')
						fout.write('graphs[1]=' + fname + '\n\n')
						if 'stats' in stats_flag and "enum" not in prob:
							fout.write(numactl_flag + './../pmc -a ' + str(alg) + ' -f ' + path + fname + ' -t ' + str(t)  + ' -s ' + ' \n\n')
						else:
							fout.write(numactl_flag + './../pmc -a ' + str(alg) + ' -f ' + path + fname + ' -t ' + str(t)  + ' \n\n')
						print shfile
						fout.close()
						sh_list.append(shfile)
	
	results_path = results_folder + '/'
	shexe = sh_path + 'run_all.sh'
	fout = open( shexe, "w+" )
	for file in sh_list:
		print file
		fout.write('sh ./' + sh_path + file + ' >& ' + results_path + '' + file.split('.')[0] + '.txt \n')
	fout.close()
	
	ensure_path_exists(results_path)
	print 'executing pmc...'
	ret = os.system('nohup sh ./' + shexe + ' >& ' + results_path + '/results_run_all.txt  &')
	
