'''
@python2.6 pmc_perf_table.py <folder> <maxlen: graphname> <regex: use graphs> <regex: ignore graph>

First, use pmc_runner.py

Then use this script to create table.
python pmc_perf_table.py results

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

def sort_types(graphs,gtypes):
	types = {}
	name2type = {}
	for t in gtypes:
		gdata = []
		for gname in graphs:
			if t in gname[0:len(t)+1]:
				#print graphs[gname]
				gdata.append(graphs[gname])
				types[t] = gdata
				name2type[gname] = t
	return types

def sort_types_auto(graphs):
	types = {}
	for gname in graphs:
		t = gname[0:2]
		val = types.get(t,'?')
		if '?' in val:
			gdata = []
			gdata.append(graphs[gname])
			types[t] = gdata
		else:
			gdata = types[t]
			gdata.append(graphs[gname])
			types[t] = gdata
	return types

def uniqueLists(g,name):
	unique = {}
	for g in graphs:
		if name in g[2]:
			m = unique.get(g[1],'f')
			if m == "f":	
				print g[1] + ' added!'
				unique[g[1]] = g
			elif m[4] < g[4]:
				print g[1] + ' switched!'
				#rm prev
				unique[m[1]] = m
	return unique

def num (s):
	try:
		return int(s)
	except ValueError:
		return format(float(s), '.1f').rstrip('0').rstrip('.')

def num_zero (s):
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
	#convert strings to ints
	for g in graphs:
		line = graphs[g]
		dt = []
		dt.append(line[0])
		for i in range(1,len(line),1):
			dt.append(num(line[i]))
		graphs[g] = dt
	return graphs

def abbr_num(n, k=1000): 
	"""Convert a number into a short string for its approximate value. 
	SI Units are used to convert the number to a human-readable range. 
	For example, abbreviateNumber(3141592) returns '3.1M'. 
	""" 
	
	k = float(k) # force floating-point divisions 
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



if len(sys.argv[1]) > 1:
	folder = sys.argv[1]


path = str(folder)
dirList=os.listdir(path)

maxlen = 13
if len(sys.argv) > 2:
	maxlen = int(sys.argv[2])

if len(sys.argv) > 3:
	dirList = [i for i in dirList if sys.argv[3] in i]
if len(sys.argv) > 4: #not
	dirList = [i for i in dirList if sys.argv[4] not in i]


#print dirList
edge_list = []
vertices = Set([])
fn_list = [];
sh_list = [];



gtypes = ['bio-', 'ca-', 'ia-', 'infra-', 'rec-', 'rt-', 'soc-', 'socfb-', 'tech-',
		 'web-', 'scc_', 'sccrt_','brock','sanr','san','p-hat','gen','keller','johnson','hamming']

graph_benchmarks = ['tscc', 'social', 'networks_small', 'output', 'gens', 'DIMACS_goldilocks', 'DIMACS_easy']

	
graphs = {}
benchmarks = {}
for fname in dirList:
	
	gstats = {}
	if "enum_" in fname:
		gstats['prob'] = 'enum'
	else:
		gstats['prob'] = 'mc'
		
	print fname
	for gtype in graph_benchmarks:
		if gtype in fname:
			gstats['graph_type'] = gtype
			fname.replace(gtype + '_','') #replace so that next is the alg name
			break

	if len(fname.split('.')[0]) > 0 and '.DS' not in fname and 'tmp' not in fname and 'results_run_all.txt' not in fname:
		line_list = open(path + '/' + fname, 'r').readlines()
		print line_list
		
		if "enum_" in fname:
			gstats['prob'] = 'enum'
		else:
			gstats['prob'] = 'mc'

		#get alg type
		if "Exact" in fname:
			gstats['alg_type'] = 'exact'
		elif "Heuristic" in fname:
			gstats['alg_type'] = 'heur'
		
		#get type of graph
		for gtype in graph_benchmarks:
			if gtype in fname:
				print gtype + " " + fname
				gstats['graph_type'] = gtype
				fname.replace(gtype + '_','') 
				break
		
		#get algorithm
		alg = fname.split('_')[0]
		gstats['alg'] = alg

		name = ''
		success = 0
		for line in line_list:
			if "File Name" in line:
				fileparts = line.split('/')
				print 'name: ' + line
				name = fileparts[len(fileparts)-1].split('.')[0]
				name = name.replace('scc_rt_','sccrt_')
				name = name.replace('infectious-dublin','infect-dublin')
				name = name.replace('infectious-hypertext2009','infect-hyper')
				gstats['name'] = name
				
			if "Size (omega): " in line:
				gstats['omega'] = int(line.split(':')[1].replace('\n','').strip())
				
			if "Time taken:" in line:
				x = float(line.split(':')[1].replace('SEC','').replace('\n','').strip())
				x_str = format(x, '.2f').rstrip('0').rstrip('.')
				if x_str == '0':
					x_str = '$<0.01$'
				gstats['time'] = x_str
				success = 1

			if "[pmc]" not in line and "graph:" not in line:
				if "[pmc: heuristic]" in line:
					gstats['mc'] = line.split(':')[2].replace('\n','').strip()
				
				if "Maximum clique:" in line and "[pmc: heuristic]" not in line:
					gstats['mc'] = line.split(':')[1].replace('\n','').strip()
					
				if "|V|:" in line:
					gstats['V'] = int(line.split(':')[1].replace('\n','').strip())
					
				if "|E|:" in line:
					gstats['E'] = int(line.split(':')[1].replace('\n','').strip())

				if "|T|:" in line:
					gstats['T'] = int(line.split(':')[1].replace('\n','').strip())
					
				if "p:" in line and not "===" in line:
					if 'e-' in line:
						idx = line.find('e-', 0, len(line))
						line = line[idx+2:len(line)].replace('\n','').strip()
						if '0' in line[0]:
							line = line[1:len(line)]
						gstats['density'] = '$10^{-' + str(line) + '}$'
					else:
						x = float(line.split(':')[1].replace('\n','').strip())
						x_str = format(x, '.3f').rstrip('0').rstrip('.')
						if x_str == '0': 
							x_str = str(line.split(':')[1].replace('\n','').strip())
							idx = x_str.find('.', 0, len(x_str))
							dec = 1
							for pos in range(idx+1,len(x_str)):
								if '0' in x_str[pos]:
									dec += 1
								else: 
									break
							gstats['density'] = '$10^{-' + str(dec) + '}$'
						else:
							gstats['density'] = x_str
					
				if "d_max:" in line:
					gstats['d_max'] = int(line.split(':')[1].replace('\n','').strip())
					
				if "d_avg:" in line:
					gstats['d_avg'] = (line.split(':')[1].replace('\n','').strip())
					
				if "T_avg:" in line:
					gstats['T_avg'] = (line.split(':')[1].replace('\n','').strip())
					
				if "T_max:" in line:
					gstats['T_max'] = int(line.split(':')[1].replace('\n','').strip())
					
				if "cc_avg:" in line:
					gstats['cc_avg'] = (line.split(':')[1].replace('\n','').strip())
				
				if "cc_global:" in line:
					gstats['cc_global'] = (line.split(':')[1].replace('\n','').strip())
					
				if "triangle assortativity:" in line:
					gstats['r_tri'] = (line.split(':')[1].replace('\n','').strip())
					
				if "K:" in line:
					gstats['K_max'] = int(line.split(':')[1].replace('\n','').strip())


				if "Heuristic found optimal solution" in line:
					gstats['omega'] = gstats['heu']
					gstats['time'] = gstats['heu_time']
					success = 1

				if "Heuristic found clique of size" in line:
					gstats['heu'] = int(line.split(' ')[5].replace('\n','').strip())

					x = float(line.split(' ')[7].replace('SEC','').replace('\n','').strip())
					x_str = format(x, '.2f').rstrip('0').rstrip('.')
					if x_str == '0':
						x_str = '$<0.01$'
					gstats['heu_time'] = x_str
				
					
		key = name + '-' + gstats['prob']
		prob = gstats['prob']
		if len(gstats) > 3 or 'enum' in prob:
			if success == 1:
				val = benchmarks.get(gstats['graph_type'], '?')
				if '?' in val:
					benchmarks[gstats['graph_type']] = {}
					graphs = benchmarks[gstats['graph_type']]
					graphs[key] = gstats
					benchmarks[gstats['graph_type']] = graphs
				else:
					graphs = benchmarks[gstats['graph_type']] #returns set of graphs of a particular type "gstats['graph_type']"
					graphs[key] = gstats 					  # add new gstats dict to graph_type dict
					benchmarks[gstats['graph_type']] = graphs
					
				graphs[key] = gstats

'''
- graphs is a list
- each graph in the list is represented by a dictionary with various properties

'''
fout = open(path + '/table_' + folder.replace('/','') + '.txt', 'w+')

gtypes_nets = ['bio-','ca-','ia-','infra-','rec-','rt-','soc-','socfb-','tech-','web-','scc_','sccrt_']
gtypes = ['bio-','ca-','ia-','infra-','rec-','rt-','soc-','socfb-','tech-','web-','scc_','sccrt_',\
		'brock','keller','MANN','c-fat','p-hat','DSJC','hamming',\
		'san','johnson','gen','C']
gtypes_dict = {'bio-':'bio', 'ca-':'collaboration', 'ia-':'interaction', \
				'infra-':'infr', 'rec-':'rec', 'rt-':'rt', 'soc-':'social', \
				'socfb-':'facebook', 'tech-':'tech', 'web-':'web-graphs',\
				 'scc_':'interaction', 'sccrt_':'retweet',\
				 'brock':'brock','keller':'keller','MANN':'MANN','c-fat':'c-fat','p-hat':'p-hat',\
				 'DSJC':'DSJC','hamming':'hamming',\
				 'san':'san','johnson':'johnson','gen':'gen',\
				 'C':'C'}
gtypes = sorted(gtypes)

for b in benchmarks:
	graphs = benchmarks[b]
	
	types = sort_types(graphs,gtypes)
	
	flag = 0
	if len(types) == 0:
		types = sort_types_auto(graphs)
		flag = 1
	
		
	types_sorted = sorted(types.keys())
	#print types_sorted
	print '\n'

	stats_order = ['name', 'V', 'E', 'T', 'density', 'd_max', 'd_avg', 'r',\
					'cc_global',\
				  'T_max', 'T_avg', 'T_ub', 'K_max', 'T_maxcore', 'omega', 'time', 'heu', 'heu_time', 'mu', 'time_mce']

	# use this dict to make header at the end before writing out the stats
	stats_dict = {'name':'\\textbf{graph}','V':'$|V|$', 'E':'$|E|$', 'T':'$|T|$',\
				 'density':'$\\rho$', 'd_max':'$d_{max}$', 'd_avg':'$d_{avg}$', \
					'cc_global':'$\\kappa$',\
					 'T_avg':'$T_{avg}$',\
					 'T_max':'$T_{max}$', 'T_ub':'$\\sqrt{2T}$', 'K_max':'$K$', 'T_maxcore':'$T_{core}$', \
					'omega':'$\\omega$', 'time':'s', 'heu':'$\\tilde{\\omega}$', 'heu_time':'$s_{heu}$', 'mu':'$\mu$', 'time_mce':'$s_{\mu}$'}
	stats_abbr = {'V':1, 'E':1,'T':1,'d_max':1,'T_max':1}
	stats_float = {'d_avg':1,'T_avg':1, 't_ub':1} #, 'time':1, 'heu_time':1}
	stats_round = {'cc_avg':1,'cc_global':1,'r':1}


	line = ''
	col_names = {}
	for t in types_sorted:
		graph_type = types[t] # collection of graphs of type t
		n_graphs = len(graph_type)
		gsort = sorted(graph_type, key=operator.itemgetter('E','name'))
		ct = 0
		if flag == 0:
			label = gtypes_dict[t]
			g = gsort[0]

			# group the data by name: so mc and enum rows for a graph are grouped together
			groups = []
			uniquekeys = []
			for k, g in groupby(gsort, lambda x: x['name']):
				groups.append(list(g))	  # Store group iterator as a list
				uniquekeys.append(k)
			
			mce_hdr = ''
			for gname in groups:
				enum_line = ''
				mc_complete = 0
				#print gname
				ncols = 0
				for g in gname: 
					val = str(g.get('prob', '?'))
					if '?' not in val and 'enum' in g['prob']:
						enum_line += ' & ' + str(g['mu']) 
						enum_line += ' & ' + str(g['time_mce'])
						col_names['mu'] = 'mu'
						col_names['time_mce'] = 'time_mce'
					elif g['E'] > 90 and 'rec-' not in t:
						
						# fixing name
						fixed_name = 0
						for type_tmp in gtypes_nets: #remove the type tag from beginning of the name
							if type_tmp in g['name']:
								g['name'] = g['name'].replace(type_tmp,'')
								name_str = g['name'].split('-')
								g['name'] = '	\\textsc{' + name_str[0][0:maxlen] + '}'
								if len(name_str) > 1 and len(name_str[0]) < maxlen:
									g['name'] += '-\\textsc{' + name_str[1][0:maxlen-len(name_str[0])] + '}'
								fixed_name = 1	
						if fixed_name == 0 and '\\textsc' not in g['name']: 
								
							name_str = g['name']
							g['name'] = '	\\textsc{' + name_str[0:maxlen] + '}'
							
						
						for col in stats_order: # systematically check if col exists, if so, append it..
							val = str(g.get(col, '?'))
							if '?' not in val:
							
								is_float = str(stats_float.get(col,'?'))
								if '?' not in is_float:
									if 'M' not in str(g[col]) and 'k' not in str(g[col]) and 'B' not in str(g[col]):
										#print col + " " + str(g[col])
										g[col] = num(g[col])
										if g[col] == 0 or len(str(g[col])) == 0:
											g[col] = '<.1'
										if len(str(g[col])) > 3 and str(g[col]) not in '<.1':
											tmp = num_zero(g[col])
											g[col] = abbr_num(int(tmp))
								
								is_round = str(stats_round.get(col,'?'))
								if '?' not in is_round:
									 g[col] = num_round(g[col])
									
								is_abbr = str(stats_abbr.get(col,'?'))
								if '?' not in is_abbr:
									if 'M' not in str(g[col]) and 'k' not in str(g[col]) and 'B' not in str(g[col]):
										if '.' in str(g[col]):
											g[col] = int(num_zero(str(g[col])))
										g[col] = abbr_num(int(g[col]))
								if 'name' in col:
									line += ' ' + str(g[col])
								else:
									line += ' & ' + str(g[col]) # + ' & '
								col_names[col] = col
						mc_complete = 1
				
				if len(enum_line) > 0 and mc_complete == 1:
					line += enum_line
				elif len(enum_line) == 0 and mc_complete == 1:
					line += ' & ' + 'X' + ' & ' + 'X'
				if mc_complete == 1:
					line += '\\\\ \n'
			if 'rec' not in t:
				line += '\\midrule \n'
			

	hdr = ''
	hdr += '\\begin{table*}[t!]\n'
	hdr += '\\caption{\\textbf{Performance on...}. }\n'
	hdr += '\\vspace{1mm}\n'
	hdr += '\\label{table:perf}\n'
	hdr += '\\centering\\small \\scriptsize\n'

	hdr += '\\begin{tabularx}{\\textwidth}{ '
	for c in stats_order:
		is_col = str(col_names.get(c,'?'))
		if '?' not in is_col:
			if 'name' in c:
				hdr += 'r '
			elif 'time' in c:
				hdr += 'r '
			elif 'omega' in c or 'heu' in c or 'mu' in c:
				hdr += 'c'
			else:
				hdr += 'X'
				
	hdr += '}\n'
	hdr += '\\toprule \n'
			

	for c in stats_order:
		is_col = str(col_names.get(c,'?'))
		if '?' not in is_col:
			if 'name' in c:
				hdr += stats_dict[c]
			else:
				hdr += ' & ' + stats_dict[c]
	hdr += mce_hdr
	
	hdr += '\\\\ \n'
	hdr += '\\midrule \n'

	bottom = '\\end{tabularx}\n' + '\\end{table*}\n'

	print hdr + line + bottom

	fout = open(path + '/table_' + folder.replace('/','') + '_' + b + '.txt', 'w+')
	fout.write(hdr + line + bottom)
	fout.close()

