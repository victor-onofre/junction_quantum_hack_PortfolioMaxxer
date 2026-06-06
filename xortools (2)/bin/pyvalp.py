#!/usr/bin/env python3

import numpy as np
import galois
import sys

def load_plinstance(filename):
    try:
        f = open(filename, 'r')
    except OSError:
        print("Could not open file: " + filename)
        return [], []
    token_array = []
    line = f.readline()
    while len(line) > 0:
        if line[0] != '#':
            token_array.append(line.split(','))
        line = f.readline()
    f.close()
    if len(token_array[0]) != 3:
        print("First line of " + filename + " is misformatted")
        print("len: " + str(len(token_array[0])))
        return [], []
    vars = int(token_array[0][0])
    cons = int(token_array[0][1])
    p = int(token_array[0][2])
    if vars < 0 or vars > 1E7:
        print("Invalid var count: " + str(vars))
        return [], []
    if cons < vars or cons > 1E8:
        print("Invalid con count: " + str(cons))
        return [], []
    if galois.is_composite(p):
        print("Invalid modulus: " + str(p) + " is not prime.")
        return [], []
    print("vars: " + str(vars) + " cons: " + str(cons) + " modulus: " + str(p))
    b = [0]*cons
    M = [ [0]*cons for _ in range(vars) ]
    for i in range(1,len(token_array)):
        bval = int(token_array[i][0])
        if bval < 0 or bval >= p:
            print("Invalid bval: " + str(bval))
            return [],[]
        b[i-1] = int(token_array[i][0])
        for j in range(1,len(token_array[i])):
            pair = token_array[i][j].split(':')
            mloc = int(pair[0])
            mval = int(pair[1])
            if mloc < 0 or mloc >= vars:
                print("Invald mloc: " + str(mloc))
                return [],[]
            if mval < 0 or mval >= p:
                print("Invalid mval: " + str(mval))
                return [],[]
            M[mloc][i-1] = mval
    GF = galois.GF(p)
    return GF(M), GF(b)

def load_psol(filename):
    try:
        f = open(filename, 'r')
    except OSError:
        print("Could not open file: " + filename)
        return [], []
    token_array = []
    line = f.readline()
    while len(line) > 0:
        if line[0] != '#':
            token_array.append(line.split(','))
        line = f.readline()
    f.close()
    if len(token_array) != 2:
        print("Too many noncomment lines in " + filename)
        return []
    if len(token_array[0]) != 2:
        print("First line is misformatted in " + filename)
        return []
    p = int(token_array[0][0])
    n = int(token_array[0][1])
    if galois.is_composite(p):
        print("Invalid modulus: " + str(p) + " is not prime.")
        return []
    if n <= 0 or n > 1E7:
        print("Invalid var count: " + str(n))
        return []
    if len(token_array[1]) != n:
        print("Vars read does not match vars claimed in" + filename)
        return []
    s = [0]*n
    for i in range(0,n):
        s[i] = int(token_array[1][i])
    GF = galois.GF(p)
    return GF(s)

#returns the number of unsatisfied equations
def evaluate_plinstance(M,b,x):
    vars, cons = M.shape
    if len(b) != cons or len(x) != vars:
        print("Dimension mismatch in evaluate_plinstance")
        print("vars: " + str(vars))
        print("cons: " + str(cons))
        print("b_length: " + str(len(b)))
        print("x_length: " + str(len(x)))
        return 0
    y = x @ M
    count = 0
    for i in range(0,len(b)):
        if y[i] != b[i]:
            count = count + 1
    return count

if len(sys.argv) != 3:
    print("usage: evalplin instance.csv varvals.psol")
    sys.exit(0)

M, b = load_plinstance(sys.argv[1])

x = load_psol(sys.argv[2])

violated = evaluate_plinstance(M,b,x)

print("violated clauses: " + str(violated))
