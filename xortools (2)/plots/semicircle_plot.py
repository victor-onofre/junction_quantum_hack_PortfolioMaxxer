import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def loadexp(filename):
    results = []
    f = open(filename, 'r')
    results=[]
    line = f.readline()
    while len(line) >= 3:
        if line[0] != '#':
            tokens = line.split()
            numread = len(tokens)
            linenums = [int(tokens[0]),int(tokens[1]),float(tokens[2])]
            results.append(linenums)
        line = f.readline()
    f.close()
    return results

results100=loadexp('exp100.txt')
results500=loadexp('exp500.txt')
results1000=loadexp('exp1000.txt')

def col(valist,index):
    rv = []
    for l in valist:
        rv.append(l[index])
    return rv

def normalize(valist, m):
    rv = []
    for entry in valist:
        nentry=[entry[0],entry[1]/m,entry[2]/m]
        rv.append(nentry)    
    return rv

nr100=normalize(results100,100)
nr500=normalize(results500,500)
nr1000=normalize(results1000,1000)

theoryx = []
theoryy = []
for lindex in range(0,1001):
    normell = lindex/1000.0
    delta = 0.5-normell
    frac=0.5+math.sqrt(0.25-delta*delta)
    theoryx.append(normell)
    if normell >= 0.5:
        theoryy.append(1.0)
    else:
        theoryy.append(frac)

matplotlib.rcParams.update({'font.size': 12})

fig1, ax1 = plt.subplots()
ax1.plot(col(nr100,1), col(nr100,2), color = 'b',label="m=100")
ax1.plot(col(nr500,1), col(nr500,2), color = 'r',label="m=500")
ax1.plot(col(nr1000,1), col(nr1000,2), color = 'g',label="m=1000")
ax1.plot(theoryx,theoryy,color='k',linestyle='dashed',label="$m \\rightarrow \infty$")
plt.xlabel("$\ell \ / \ m$")
plt.ylabel(r'$\langle s \rangle / m$')
plt.gca().set_aspect('equal')
ax1.legend()
ax1.set_xlim([0, 0.5])
ax1.set_ylim([0.5,1.0])
fig1.savefig("semicircle3.pdf")