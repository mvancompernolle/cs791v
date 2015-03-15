'''
@author: Ryan Rossi

This script creates the perf table and data to be plotted

python2.6 pmc_perf_plot.py -p path -f outfile -a algs_to_use -d discard_graphs

Example:  
python2.6 pmc_perf_plot.py -p results -f mc_perf_plot.txt

'''

__author__      = "Ryan A. Rossi"
__copyright__   = "Copyright 2012-2013, Ryan A. Rossi"
__email__ 		= "rrossi@purdue.edu"
__version__ 	= "1.0"

import sys
import os
from sets import Set
import operator
from operator import itemgetter, attrgetter
import pmc_utils
from itertools import groupby
from optparse import OptionParser
import cmath
import math


def sort_types(graphs, gtypes):
	types = {}
	name2type = {}
	for t in gtypes:
		gdata = []
		for gname in graphs:
			if t in gname[0:len(t) + 1]:
				gdata.append(graphs[gname])
				types[t] = gdata
				name2type[gname] = t
	return types

def sort_types_auto(graphs):
	types = {}
	for gname in graphs:
		t = gname[0:2]
		val = types.get(t, '?')
		if '?' in val:
			gdata = []
			gdata.append(graphs[gname])
			types[t] = gdata
		else:
			gdata = types[t]
			gdata.append(graphs[gname])
			types[t] = gdata
	return types

def uniqueLists(g, name):
	unique = {}
	for g in graphs:
		if name in g[2]:
			m = unique.get(g[1], 'f')
			if m == "f":	
				print g[1] + ' added!'
				unique[g[1]] = g
			elif m[4] < g[4]:
				print g[1] + ' switched!'
				# rm prev
				unique[m[1]] = m
	return unique

def num (s):
	try:
		return int(s)
	except ValueError:
		return format(float(s), '.0f').rstrip('0').rstrip('.')

def num_round (s):
	try:
		return int(s)
	except ValueError:
		return format(float(s), '.2f').rstrip('0').rstrip('.')
		
def convert_nums(graphs):
	# convert strings to ints
	for g in graphs:
		line = graphs[g]
		dt = []
		dt.append(line[0])
		for i in range(1, len(line), 1):
			dt.append(num(line[i]))
		graphs[g] = dt
	return graphs

def abbr_num(n, k=1000): 
	k = float(k)  # force floating-point divisions 
	nk = 0 
	sign = '' 
	if n < 0: 
		n = -n 
		sign = '-' 
	while n >= 999.5: 
		n /= k 
		nk += 1 
	while 0 < n < 0.9995: 
		n *= k 
		nk -= 1 
	try: 
		if nk > 0:
			suffix = ' kMBTPEZY'[nk] 
		elif nk < 0: 
			suffix = ' munpfazy'[-nk] 
		else: suffix = '' 
	except: 
		suffix = 'k^%d' % nk 

	if n < 9.95 and n != round(n): 
		strn = "%1.1f" % n 
	else: 
		strn = "%d" % round(n) 

	return sign + strn + suffix 

def entropy(freqList):
	ent = 0.0
	for freq in freqList:
		if freq > 0:
			ent = ent + freq * math.log(freq, 2)
		elif freq == 0:
			ent = ent + 0
	ent = -ent
	print int(math.ceil(ent))
	return int(math.ceil(ent))

def fixup_graph_name(fname):
	name = fname.replace('-pruning','')
	name = name.replace('.txt','')
	return name


def read_baseline_perf(baseline_folder):
	dirList=os.listdir(baseline_folder)
	prefix = 'baseline'
	base_perf = []
	for filename in dirList:
		gstats = {}
		if 'sample_social' in filename:
			gname = filename.replace(prefix,'').replace('.txt','')
			gstats['graph_name'] = gname
			# open file and read/store the lines for processing
			line_list = open(baseline_folder + '/' + filename, 'r').readlines()
			tmp = ''
			for line in line_list:
				tmp += line
			# get each time and store pruning info if containing it
			line = tmp.strip()
			if 'largest cliques:' in line:
				mcdata = line.split(':')[1].strip()
				perf = mcdata.split(',')
				time = float(perf[1].strip())
				print filename + ', ' + gname + ', ' + str(time)
			else: 
				print ''
				print '*** ' + gname + ' did not terminate!\n'
				time = NaN
			gstats['prob'] = 'mc'
			gstats['name'] = gname
			gstats['filename'] = filename 
			gstats['alg'] = 'base'
			gstats['threads'] = 't1'
			gstats['time'] =  time
			base_perf.append(gstats)
			print gstats
	return base_perf
				
			

# this func can be customized with other rules for parsing
def get_pruning_info(line, gstats, verbose):
	
	line = line.strip()
	
	if "File Name" in line:
		fileparts = line.split('/')
		name = fileparts[len(fileparts)-1].split('.')[0]
		name = name.replace('scc_rt_','sccrt_')
		name = name.replace('infectious-dublin','infect-dublin')
		name = name.replace('infectious-hypertext2009','infect-hyper')
		gstats['graph_name'] = name

	if "Size (omega): " in line:
		gstats['omega'] = int(line.split(':')[1].replace('\n','').strip())
		
	if "Time taken:" in line:
		x = float(line.split(':')[1].replace('SEC','').replace('\n','').strip())
	#				x_str = format(x, '.2f').rstrip('0').rstrip('.')
		gstats['time'] = x # don't round
	
	if "[pmc]" not in line and "graph:" not in line:
			
		if "|V|:" in line:
			gstats['V'] = int(line.split(':')[1].replace('\n','').strip())
			
		if "|E|:" in line:
			gstats['E'] = int(line.split(':')[1].replace('\n','').strip())
			
		if "K:" in line:
			gstats['K_max'] = int(line.split(':')[1].replace('\n','').strip())
	
		if "Heuristic found optimal solution" in line:
			gstats['omega'] = gstats['heu']
			gstats['time'] = gstats['heu_time']
			gstats['heu_optimal'] = 1
	
		if "Heuristic found clique of size" in line:
			gstats['heu'] = int(line.split(' ')[5].replace('\n','').strip())
			x = float(line.split(' ')[7].replace('SEC','').replace('\n','').strip())
			gstats['heu_time'] = x # don't round
	
	return gstats


def get_data(graph_data, pdata, values_to_keep, prob):
	for val in values_to_keep: # get all values and store them in pdata (i.e., pdata['alg'] has alg4)
		if ':' in val:
			val_pair = val.split(':')
			key_to_key = val_pair[0]					# key_to_key = alg
			key_to_value = val_pair[1]					# key_to_value = time
			
			if key_to_key in graph_data and key_to_value in graph_data:
				saved_key = graph_data[key_to_key]			# this is the key
			
				saved_val = graph_data[key_to_value]		# this will be the value
				print saved_key
				print saved_val
				pdata[saved_key] = saved_val    # add all such values to the dict
							
			else:
				prob.append(graph_data['graph_name'] + ':' + graph_data['threads'] + ':' + graph_data['alg'] )


	return [pdata, prob]


def get_graph_dict(P,use_key,values_to_keep):
	problems = []
	perf_data = {}
	for g in P:
		print g
		
		# make key
		gkey = ''
		for k in use_key:
			if k in g:
				tmp = str(g[k])
				if len(gkey) == 0:
					gkey = tmp
				else:
					gkey = gkey + ':' + tmp
		print gkey
		
		if gkey in perf_data:    	   # check if key already exists in perf dictionary
			pdata = perf_data[gkey]
			
			[pdata, problems] = get_data(g, pdata, values_to_keep, problems)
			perf_data[gkey] = pdata	   # store the pdata back into the dict
		else: 						   # first time gkey encountered! add it!
			pdata = {}
			[pdata, problems] = get_data(g, pdata, values_to_keep, problems)
			perf_data[gkey] = pdata
		print perf_data
	print '\nproblematic graphs/runs:'
	print problems
	return perf_data

def main():

	parser = OptionParser()
	parser.add_option("-f", "--file", dest="filename",
	                  help="write table to FILE", metavar="FILE")
	parser.add_option("-p", "--path", dest="path",
	                  help="read files in directory from PATH", metavar="PATH")
	parser.add_option("-d", "--disc", dest="discard", default='',
	                  help="discard files with substring DISCARD", metavar="DISC")
	parser.add_option("-a", "--algs", dest="algs", default='4',
	                  help="algorithms to include in the table: '4,3'", metavar="ALGS")
	parser.add_option("-q", "--quiet",
	                  action="store_false", dest="verbose", default=False,
	                  help="don't print status messages to stdout")
	(opts, args) = parser.parse_args()
	print args
	print opts
	
	verbose = opts.verbose
	
	baseline_folder = 'baseline_results/'
	base_perf = read_baseline_perf(baseline_folder)
	
	# form a list from the algs
	alg_list = opts.algs.split(',')
	print alg_list
	
	
	# get the files in the directory given by path
	dirList=os.listdir(opts.path)
	
	# ignore files with a particular substring 
	if len(opts.discard) > 0: 
		dirList = [i for i in dirList if opts.discard not in i]
		
	print 'FILES LOCATED IN THE DIRECTORY: ' + opts.path
	print dirList
	
	
	# P is a list of dictionaries: [{},{},{}]
	P = []
	# process each file one by one
	for fname in dirList:
		
		# key is a graph name --> value is a list
		gstats = {}
		
		# skip these files
		if len(fname.split('.')[0]) == 0 or '.DS' in fname or 'tmp' in fname or 'results_run_all.txt' in fname:
			continue
	
		# Name Example: type_alg_mc_t1_soc-flickr.txt
		# type[output,tscc]_alg[0,1,2,4]_[mc/enum]_t[0,1,2,4,8,16]
		fn = fname
		fn = fn.replace('sample_graphs','sample-graphs')
		fn = fn.replace('sample_social','sample-social')
		fn = fn.replace('sample_benchmark','sample-benchmark')
		
		# get graph info from FILENAME for categorization
		parts = fn.split('_')
		gstats['type'] = parts[0]
		gstats['alg'] = parts[1]#.replace('alg','')
		gstats['prob'] = parts[2]
		gstats['threads'] = parts[3]#.replace('t','')
		gstats['name'] = parts[4]
		gstats['filename'] = fname # store original here
		
		if verbose: print 'processing ' + gstats['name']
	
		# open file and read/store the lines for processing
		line_list = open(opts.path + '/' + fname, 'r').readlines()
		
		# get each time and store pruning info if containing it
		for line in line_list:
			gstats = get_pruning_info(line, gstats, verbose)
	
		# verify gstats, then add to pruning dict P
		P.append(gstats)

	P = P + base_perf
	# define the key here, i.e., 'soc-flickr:t16'
	use_key = ['graph_name','threads']
	# define the values to keep
	values_to_keep = ['alg:time'] # means: resulting_dict[ key: dict[alg] ] := [ value: dict[time] ]
	gperf_data = get_graph_dict(P,use_key,values_to_keep)
	
	print gperf_data.keys()
	
	name_list = []
	
	# get unique names and threads
	unique_names = Set([])
	unique_threads = Set([])
	for g in gperf_data:
		key_parts = g.split(':')
		unique_names.add(key_parts[0])
		unique_threads.add(key_parts[1])
		gp = gperf_data[g]
	print '\n'
	print unique_names
	print '\n'
	print unique_threads
	
	# get unique algs
	unique_algs = Set([])
	for g in P:
		unique_algs.add(g['alg'])
		
	algs_sorted = sorted(unique_algs)
	
	
	names_sorted = sorted(unique_names)
	print names_sorted
	
	
	print unique_algs
	unique_algs.discard('alg2')
	print unique_algs
	
	for t in unique_threads:
		fout_name = 'plots/mcdata/' + opts.filename + '_threads_' + t + '.txt'
		print '\n'
		print 'threads data saved: ' + fout_name
		fout = open(fout_name, 'w+')
		gnames_out = ''
		for gname in unique_names:
			gnames_out += gname + ':' + t + '\n'
			key = gname + ':' + t
			if key in gperf_data:
				perf = gperf_data[key]
				perf_str = ''
				alg_str = ''
				for alg in unique_algs:
					if alg in perf:
						alg_str += alg + ' '
						perf_str += str(perf[alg]) + '\t'
					else:
						print key
						perf_str += 'NaN' + '\t'
			fout.write(perf_str + '\n')
			print perf_str
		fout.close()
		
	print unique_algs
	print unique_threads
	
	
	fout_labels = 'plots/mcdata/' + opts.filename + '_threads_names' + '.txt'
	print 'data graph names saved: ' + fout_labels
	fout = open(fout_labels, 'w+')
	fout.write(gnames_out + '\n')
	fout.write(alg_str + '\n')
	fout.close()	
	
	
	# write out also the info to make the plot: |V|+|E| by runtime
	fout = open('plots/mcdata/' + opts.filename + '_runtime_VE_plot.txt', 'w+')
	for g in P:
		if g['threads'] == 't1' and g['alg'] == 'alg0':
			if 'time' in g:
				total = g['E'] + g['V']
				runtime = g['time']
				name = g['name']
				fout.write(name + '\t' + str(total) + '\t' + str(runtime) + '\n')
	fout.close()

	

if __name__ == "__main__":
    main()