#!/usr/bin/env python3

import numpy as np
import random
import sys
import ldpc
from ldpc import bp_decoder
from ldpc.code_util import construct_generator_matrix

def load_instance(filename):
    try:
        f = open(filename, 'r')
    except OSError:
        print("Could not open file: " + filename)
        return [], [], []
    numread = 1
    firstline = True
    var_index = 0
    con_index = 0
    val_index = 0
    line = f.readline()
    while len(line) > 0:
        conline = False
        if line[0] != '#':
            tokens = line.split()
            numread = len(tokens)
            if not firstline and con_index < num_cons:
                conline = True
                for i in range(len(tokens)):
                    var_index = int(tokens[i])
                    c[con_index].append(var_index)
                    v[var_index].append(con_index)
                con_index = con_index + 1
            if not firstline and not conline and val_index < num_cons:
                if len(tokens) != 1:
                    print("Warning: Formatting error in value line")
                    print(line)
                else:
                    b[val_index] = int(tokens[0])
                    val_index = val_index + 1
            if firstline:
                if numread != 2:
                    print("Error first line must have two tokens")
                    return [],[],[]
                num_vars = int(tokens[0])
                num_cons = int(tokens[1])
                firstline = False
                if num_vars < 0 or num_vars > 1E6:
                    print("num_vars(" + str(num_vars) + ") is out of range.")
                    return [],[],[]
                if num_cons < num_vars or num_cons > 1E6:
                    print("num_cons(" + str(num_cons) + ")  is out of range.")
                    return [],[],[]
                v = [ [] for _ in range(num_vars) ]
                c = [ [] for _ in range(num_cons) ]
                b = [ 0 for _ in range(num_cons) ]
        line = f.readline()
    if con_index != val_index:
        print("Warning, number of values doesn't match number of constraints!")
    f.close()
    return v,c,b

def load_weighted_instance(filename):
    try:
        f = open(filename, 'r')
    except OSError:
        print("Could not open file: " + filename)
        return [], [], []
    numread = 1
    firstline = True
    var_index = 0
    con_index = 0
    line = f.readline()
    while len(line) > 0:
        conline = False
        if line[0] != '#':
            tokens = line.split()
            numread = len(tokens)
            if not firstline and con_index < num_cons:
                conline = True
                w[con_index] = float(tokens[0])
                for i in range(1,len(tokens)):
                    var_index = int(tokens[i])
                    c[con_index].append(var_index)
                    v[var_index].append(con_index)
                con_index = con_index + 1
            if firstline:
                if numread != 2:
                    print("Error first line must have two tokens")
                    return [],[],[]
                num_vars = int(tokens[0])
                num_cons = int(tokens[1])
                firstline = False
                if num_vars < 0 or num_vars > 1E6:
                    print("num_vars(" + str(num_vars) + ") is out of range.")
                    return [],[],[]
                if num_cons < num_vars or num_cons > 1E6:
                    print("num_cons(" + str(num_cons) + ")  is out of range.")
                    return [],[],[]
                v = [ [] for _ in range(num_vars) ]
                c = [ [] for _ in range(num_cons) ]
                w = [ 0.0 for _ in range(num_cons) ]
        line = f.readline()
    f.close()
    return v,c,w

def parity_check(v,c):
    H = np.zeros((len(c), len(v)))
    for con in range(len(c)):
        for var in c[con]:
            H[con, var] = 1
    return np.transpose(H)

def rand_word(G):
    (logical,physical) = G.shape
    generator = np.zeros(logical)
    for i in range(logical):
        generator[i] = str(random.randint(0, 1))
    return np.matmul(generator,G)%2

def rand_err(n,e):
    all_bits = range(n)
    error_bits = random.sample(all_bits,e)
    x = np.zeros(n)
    for i in error_bits:
        x[i] = 1
    return x

def write_list(f, a):
    n = len(a)
    for i in range(n):
        f.write(str(a[i]))
        if(i < n-1):
            f.write(", ")
        else:
            f.write('\n')

if(len(sys.argv) != 2):
    print("usage: load_decode_jr.py instance.tsv")
    sys.exit(0)

v, c, w = load_weighted_instance(sys.argv[1])

#print(v)
#print(c)
#print(w)

if len(v) == 0:
    print("Unable to load instance " + str(sys.argv[1]))
    sys.exit()

outname = sys.argv[1] + ".out"
f = open(outname, 'w')

H = parity_check(v, c)
#G = construct_generator_matrix(H)
#n = G.shape[1]
n = H.shape[1]

def emax(osdmethod):
    emax = 0
    errors = 0
    rates = []
    rate = 1
    while errors < n and rate >= 0.8:
        if osdmethod == "none":
            bpd = bp_decoder(
                H, #the parity check matrix
                error_rate=errors/n, # the error rate on each bit
                max_iter=n, #the maximum iteration depth for BP
                bp_method="product_sum", #BP method. The other option is `minimum_sum'
                channel_probs=[None] #channel probability probabilities. Will overide error rate.
            )
        else:
            bpd = ldpc.osd.bposd_decoder(
                H, #the parity check matrix
                error_rate=errors/n, # the error rate on each bit
                max_iter=n, #the maximum iteration depth for BP
                bp_method="product_sum", #BP method. The other option is `minimum_sum'
                osd_method=osdmethod,
                channel_probs=[None] #channel probability probabilities. Will overide error rate.
            )
        success = 0
        for repetition in range(100):
            #r=rand_word(G)
            e=rand_err(n,errors)
            syndrome=H@e%2
            #corrupted=(r+e)%2
            inferred_error=bpd.decode(syndrome)
            if np.array_equal(e, inferred_error, equal_nan=False) == True:
                success = success + 1
        rate = success/100
        print(str(errors) + ": " + str(rate))
        rates.append(rate)
        if rate > 0.9:
            emax = errors
        errors = errors + 1
    write_list(f,rates)
    print("emax (osd = " + osdmethod +"): " + str(emax))

emax("none")
emax("osd_0")
emax("osd_e")
emax("osd_cs")
