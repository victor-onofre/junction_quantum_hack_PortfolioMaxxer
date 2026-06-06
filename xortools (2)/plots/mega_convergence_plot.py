import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

f = open('mega_summary.txt', 'r')
tokenarray=[]
line = f.readline()
while len(line) > 0:
    if line[0] != '#':
        line = line.replace("," , " ") #handle both tsv and csv
        line = line.replace(":" , " ")
        tokens = line.split()
        numread = len(tokens)
        tokenarray.append(tokens)
    line = f.readline()
f.close()

name = []
sat = []
sweeps = []
satfrac = []
for line in tokenarray:
    namestring = line[0]
    namestring = namestring.replace("_"," ")
    namestring = namestring.replace("."," ")
    tokens = namestring.split()
    sweepcount = int(tokens[1])
    name.append(line[0])
    sweeps.append(sweepcount)
    sat.append(int(line[7]))
    satfrac.append(int(line[7])/50000)

fig1, ax1 = plt.subplots()
ax1.scatter(sweeps, satfrac, color = 'b', marker='.', label="Simulated Annealing")
plt.axhline(y=41550/50000, color='r', linestyle='--', label="DQI+BP")
#plt.axhline(y=41700/50000, color='orange', linestyle='--', label="DQI+BP modulo conjectures")
ax1.legend()
plt.xlabel("sweeps")
plt.ylabel("fraction satisfied")
fig1.savefig('mega_convergence.pdf',bbox_inches="tight")