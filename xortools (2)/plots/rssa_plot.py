import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

f = open("rscombined.txt", 'r')
tokenarray=[]
line = f.readline()
while len(line) > 0:
    if line[0] != '#':
        line = line.replace("," , " ") #handle both tsv and csv
        tokens = line.split()
        numread = len(tokens)
        tokenarray.append(tokens)
    line = f.readline()
f.close()
mdata = []
ndata = []
sdata = []
for line in tokenarray:
    if len(line) != 6:
        print("uh oh: ")
        print(line)
    else:
        name = line[0]
        name = name.replace("log","")
        name = name.replace("rs","")
        name = name.replace("_"," ")
        name = name.replace("."," ")
        name = name.replace(":", " ")
        nametokens = name.split()
        n = int(nametokens[0])
        m = int(nametokens[1])
        sat = int(nametokens[2])
        ndata.append(n)
        mdata.append(m)
        sdata.append(sat)

phidata = []
rdata = []
rcat = []
for i in range(0,len(mdata)):
    phi = sdata[i]/mdata[i] - 0.5
    phidata.append(phi)
    r = mdata[i]/ndata[i]
    rdata.append(r)
    rcat.append(round(r))

mdata2 = []
ndata2 = []
sdata2 = []
phidata2 = []
rdata2 = []
mdata6 = []
ndata6 = []
sdata6 = []
phidata6 = []
rdata6 = []
mdata10 = []
ndata10 = []
sdata10 = []
phidata10 = []
rdata10 = []
for i in range(0,len(mdata)):
    if rcat[i] == 2:
        mdata2.append(mdata[i])
        ndata2.append(ndata[i])
        sdata2.append(sdata[i])
        phidata2.append(phidata[i])
        rdata2.append(rdata[i])
    if rcat[i] == 6:
        mdata6.append(mdata[i])
        ndata6.append(ndata[i])
        sdata6.append(sdata[i])
        phidata6.append(phidata[i])
        rdata6.append(rdata[i])
    if rcat[i] == 10:
        mdata10.append(mdata[i])
        ndata10.append(ndata[i])
        sdata10.append(sdata[i])
        phidata10.append(phidata[i])
        rdata10.append(rdata[i])

logm = []
logphi = []
for i in range(0,len(mdata)):
    logm.append(np.log(mdata[i]))
    logphi.append(np.log(phidata[i]))
coef=np.polyfit(logm,logphi,1)
print(coef)

fitm = [min(mdata),max(mdata)]
fitphi = []
for i in range(0,len(fitm)):
    fitphi.append(math.exp(coef[1] + coef[0]*math.log(fitm[i])))

matplotlib.rcParams.update({'font.size': 12})
fig1, ax1 = plt.subplots()
ax1.scatter(mdata2, phidata2, marker='.', color='g', label="m/n = 2")
ax1.scatter(mdata6, phidata6, marker='.', color='b', label="m/n = 6")
ax1.scatter(mdata10, phidata10, marker='.', color='r', label="m/n = 10")
ax1.plot(fitm,fitphi,color='k',label="fit")
ax1.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("m (log scale)")
plt.ylabel("$\phi_{\max}}$ $-$ $1/2$ (log scale)")
fig1.tight_layout()
fig1.savefig("rs_sa2.pdf")