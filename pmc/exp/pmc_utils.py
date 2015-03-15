__author__      = "Ryan A. Rossi"
__copyright__   = "Copyright 2012-2013, Ryan A. Rossi"
__email__ 	= "rrossi@purdue.edu"
__version__ 	= "1.0"

import sys
import os
import datetime as dt
import time
import subprocess
import operator
from operator import itemgetter, attrgetter

def num (s):
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
            suffix = ' kMGTPEZY'[nk] 
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

def ensure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise