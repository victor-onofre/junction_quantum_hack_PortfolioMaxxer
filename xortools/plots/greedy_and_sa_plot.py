import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def extract_column(numtable, index):
    rows = len(numtable)
    cols = len(numtable[0])
    returnval = []
    if rows < 2:
        print("Nobody plots one data point")
        return returnval
    if index >= cols or index < 0:
        print("Invalid column index")
        return returnval
    for row in numtable:
        returnval.append(row[index])
    return returnval

def first(elem):
    return elem[0]

def plotit(input, output, plottitle):
    f = open(input, 'r')
    newtokenarray=[]
    line = f.readline()
    while len(line) > 0:
        if line[0] != '#':
            line = line.replace("," , " ") #handle both tsv and csv
            tokens = line.split()
            numread = len(tokens)
            newtokenarray.append(tokens)
        line = f.readline()
    f.close()

    results = []
    for line in newtokenarray:
        if len(line) != 6:
            print("uh oh: ")
            print(line)
        else:
            name = line[0]
            name = name.replace("log","")
            name = name.replace("l","")
            name = name.replace("_"," ")
            name = name.replace("."," ")
            name = name.replace(":", " ")
            nametokens = name.split()
            k = int(nametokens[0])
            D = int(nametokens[1])
            bsize = int(nametokens[2])
            sat = int(nametokens[3])
            m = int(line[5])
            result = [k,D,bsize,sat,m]
            results.append(result)

    halfD =[]
    halfE = []
    halfLogD = []
    halfLogE = []
    nineD = []
    nineE = []
    nineLogD = []
    nineLogE = []
    oneD = []
    oneE = []
    oneLogD = []
    oneLogE = []
    for r in results:
        thisK = r[0]
        thisD = r[1]
        thisBSIZE = r[2]
        thisSAT = r[3]
        thisM = r[4]
        thisFRAC = thisSAT / thisM
        thisE = thisFRAC - 0.5
        if abs(thisK/thisD - 0.5) < 0.05:
            halfD.append(thisD)
            halfE.append(thisE)
            halfLogD.append(np.log(thisD))
            halfLogE.append(np.log(thisE))
        if abs(thisK/thisD - 0.9) < 0.05:
            nineD.append(thisD)
            nineE.append(thisE)
            nineLogD.append(np.log(thisD))
            nineLogE.append(np.log(thisE))
        if abs(thisK/thisD - 0.1) < 0.05:
            oneD.append(thisD)
            oneE.append(thisE)
            oneLogD.append(np.log(thisD))
            oneLogE.append(np.log(thisE))

    halfcoef=np.polyfit(halfLogD,halfLogE,1)
    ninecoef=np.polyfit(nineLogD,nineLogE,1)
    onecoef=np.polyfit(oneLogD,oneLogE,1)

    print("Fit coefficients:")
    print(halfcoef)
    print(ninecoef)
    print(onecoef)

    exponent = onecoef[0]
    onelabel = rf"$\phi_{{\max}} \propto D^{{{exponent:.2f}}}$"
    exponent = halfcoef[0]
    halflabel = rf"$\phi_{{\max}} \propto D^{{{exponent:.2f}}}$"
    exponent = ninecoef[0]
    ninelabel = rf"$\phi_{{\max}} \propto D^{{{exponent:.2f}}}$"

    matplotlib.rcParams.update({'font.size': 12})
    halffitdata = []
    for i in range(0,len(halfD)):
        halffitdata.append([halfD[i],math.exp(halfcoef[1] + halfcoef[0]*math.log(halfD[i]))])
    halffitdata.sort(key=first)

    onefitdata = []
    for i in range(0,len(oneD)):
        onefitdata.append([oneD[i],math.exp(onecoef[1] + onecoef[0]*math.log(oneD[i]))])
    onefitdata.sort(key=first)

    ninefitdata = []
    for i in range(0,len(nineD)):
        ninefitdata.append([nineD[i],math.exp(ninecoef[1] + ninecoef[0]*math.log(nineD[i]))])
    ninefitdata.sort(key=first)

    fig, ax = plt.subplots()
    ax.set(title=plottitle)
    ax.set(xscale="log")
    ax.set(yscale="log")
    ax.set_yticks([2E-2,4E-2,8E-2,1.6E-1])
    ax.set_yticklabels(["$2 \\times 10^{-2}$","$4 \\times 10^{-2}$","$8 \\times 10^{-2}$","$1.6 \\times 10^{-1}$"])
    plt.scatter(oneD, oneE, label="k/D = 0.1")
    plt.scatter(halfD, halfE , label="k/D = 0.5")
    plt.scatter(nineD, nineE, label="k/D=0.9")
    plt.plot(extract_column(onefitdata,0),extract_column(onefitdata,1),label=onelabel)
    plt.plot(extract_column(halffitdata,0),extract_column(halffitdata,1),label=halflabel)
    plt.plot(extract_column(ninefitdata,0),extract_column(ninefitdata,1),label=ninelabel)
    plt.xlabel("D")
    plt.ylabel("$\phi_{\max}}$ $-$ $1/2$")
    ax.legend()
    fig.savefig(output, bbox_inches="tight")

plotit("bdgr_combined.txt", "greedyfits2.pdf", "Greedy Algorithm")

plotit("bd_combined.txt", "SAfits2.pdf", "Simulated Annealing")