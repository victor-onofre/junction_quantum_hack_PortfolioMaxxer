#!/usr/bin/env python3

import os
import sys

def fulljob(k,D,bsize):
    labelstring = str(k) + '_' + str(D) + '_' + str(bsize)
    logname = 'rrlog' + labelstring + '.txt'
    os.system('random_regular ' + str(k) + ' ' + str(D) + ' ' + str(bsize) + ' > rrlog' + labelstring + '.txt')
    instancename = 'mx' + labelstring + '.tsv'
    os.system('load_decode ' + instancename + ' > ldlog' + labelstring + '.txt')
    fp = open('ldlog' + labelstring + '.txt')
    lines = fp.readlines()
    fp.close()
    lastline = lines[len(lines)-1]
    tokens = lastline.split()
    emax = int(tokens[2])
    os.system('epsilon ' + str(emax) + ' ' + instancename + ' > eplog' + labelstring + '.txt')
    fp = open('eplog' + labelstring + '.txt')
    lines = fp.readlines()
    fp.close()
    for line in lines:
        tokens = line.split()
        if tokens[0] == 'median_mag:':
            median_mag = float(tokens[1])
    os.system('sawxor ' + instancename + ' > salog' + labelstring + '.txt')
    fp = open('salog' + labelstring + '.txt')
    lines = fp.readlines()
    fp.close()
    lastline = lines[len(lines)-1]
    tokens = lastline.split()
    best_val = float(tokens[4])
    n = k * bsize
    m = D * bsize
    results_string = instancename + '\t' + str(k) + '\t' + str(D) + '\t' + str(n) + '\t' + str(m)
    results_string = results_string + '\t' + str(median_mag) + '\t' + str(abs(best_val))
    if median_mag > abs(best_val):
        results_string = results_string + '\twin!\n'
    else:
        results_string = results_string + '\n'
    fp = open("biglog1.txt", "a")  # append mode
    fp.write(results_string)
    fp.close()

if len(sys.argv) != 4 and len(sys.argv) != 5:
    print("usage: fulljob k D bsize <working dir>")
    sys.exit(0)

if sys.argv == 5:
    os.chdir(sys.argv[4])

k = int(sys.argv[1])
D = int(sys.argv[2])
bsize = int(sys.argv[3])

if bsize < 1 or k < 2 or D <= k:
    print("invalid parameters")
    sys.exit(0)

fulljob(k,D,bsize)
