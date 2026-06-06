import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def loadlog(filename):
    tokenarray = []
    f = open(filename, 'r')
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
    return tokenarray

tokenarray = loadlog("arn_ar_combined_new.txt")
arn_ar_Dlist = []
arn_ar_phiup = []
for line in tokenarray:
    if len(line) == 7:
        name = line[0]
        satisfied = int(line[1])
        m = int(line[6])
        name = name.replace("/"," ")
        name = name.replace("lar", "")
        name = name.replace("_", " ")
        name = name.replace("."," ")
        nametokens = name.split()
        k = int(nametokens[0])
        D = int(nametokens[1])
        b = int(nametokens[2])
        arn_ar_Dlist.append(D)
        arn_ar_phiup.append(satisfied/m - 1/2)

tokenarray=loadlog('arn_sa_combined_new.txt')
arn_sa_Dlist = []
arn_sa_phiup = []
for line in tokenarray:
    if len(line) == 7:
        name = line[0]
        satisfied = int(line[1])
        m = int(line[6])
        name = name.replace("/"," ")
        name = name.replace("lsa", "")
        name = name.replace("_", " ")
        name = name.replace("."," ")
        nametokens = name.split()
        k = int(nametokens[0])
        D = int(nametokens[1])
        b = int(nametokens[2])
        arn_sa_Dlist.append(D)
        arn_sa_phiup.append(satisfied/m - 1/2)

tokenarray = loadlog('arn_dqi_combined_new.txt')
arn_dqi_Dlist = []
arn_dqi_emaxlist = []
arn_dqi_mlist = []
for line in tokenarray:
    if len(line) == 4:
        name = line[0]
        emax = int(line[3])
        name = name.replace("ldq", "")
        name = name.replace("_", " ")
        name = name.replace("."," ")
        nametokens = name.split()
        k = int(nametokens[0])
        D = int(nametokens[1])
        b = int(nametokens[2])
        if emax != 0:
            arn_dqi_Dlist.append(D)
            arn_dqi_emaxlist.append(emax)
            arn_dqi_mlist.append(D*b)

arn_dqisat=[]
arn_dqi_phiup=[]
for i in range(len(arn_dqi_mlist)):
    delta = 0.5-arn_dqi_emaxlist[i]/arn_dqi_mlist[i]
    sat = arn_dqi_mlist[i] * (0.5 + math.sqrt(0.25 - delta*delta))
    arn_dqisat.append(sat)
    arn_dqi_phiup.append(math.sqrt(0.25 - delta*delta))

arn_logsaD = []
arn_logsae = []
arn_logdqD = []
arn_logdqe = []
arn_logarD = []
arn_logare = []
for i in range(0,len(arn_sa_Dlist)):
    arn_logsaD.append(np.log(arn_sa_Dlist[i]))
    arn_logsae.append(np.log(arn_sa_phiup[i]))
for i in range(0, len(arn_dqi_Dlist)):
    arn_logdqD.append(np.log(arn_dqi_Dlist[i]))
    arn_logdqe.append(np.log(arn_dqi_phiup[i]))
for i in range(0, len(arn_ar_Dlist)):
    arn_logarD.append(np.log(arn_ar_Dlist[i]))
    arn_logare.append(np.log(arn_ar_phiup[i]))


print("fit coefficients")

arn_sa_coef=np.polyfit(arn_logsaD,arn_logsae,1)
print(arn_sa_coef)

arn_ar_coef=np.polyfit(arn_logarD,arn_logare,1)
print(arn_ar_coef)

arn_dqi_coef=np.polyfit(arn_logdqD,arn_logdqe,1)
print(arn_dqi_coef)

arn_sa_fitdata = []
for i in range(0,len(arn_sa_Dlist)):
    arn_sa_fitdata.append(math.exp(arn_sa_coef[1] + arn_sa_coef[0]*math.log(arn_sa_Dlist[i])))

arn_ar_fitdata = []
for i in range(0,len(arn_ar_Dlist)):
    arn_ar_fitdata.append(math.exp(arn_ar_coef[1] + arn_ar_coef[0]*math.log(arn_ar_Dlist[i])))

arn_dqi_fitdata = []
for i in range(0,len(arn_dqi_Dlist)):
    arn_dqi_fitdata.append(math.exp(arn_dqi_coef[1] + arn_dqi_coef[0]*math.log(arn_dqi_Dlist[i])))

arn_qaoa_Dlist = np.sort(arn_sa_Dlist)
arn_qaoadata = []
for i in range(0,len(arn_qaoa_Dlist)):
    arn_qaoadata.append(0.6422*math.sqrt(3/(2*(arn_qaoa_Dlist[i]-1))))

matplotlib.rcParams.update({'font.size': 12})
fig1, ax1 = plt.subplots()
ax1.scatter(arn_ar_Dlist, arn_ar_phiup, color = 'b', marker='x',label="AdvRand")
ax1.plot(arn_ar_Dlist,arn_ar_fitdata,color='b',label="$0.31 D^{-0.49}$")
ax1.scatter(arn_sa_Dlist, arn_sa_phiup, color = 'r', marker='x',label="Simulated Annealing")
ax1.plot(arn_sa_Dlist,arn_sa_fitdata,color='r',label="$0.90 D^{-0.49}$")
ax1.scatter(arn_dqi_Dlist, arn_dqi_phiup, color = 'g', marker='x',label="DQI+BP")
ax1.plot(arn_dqi_Dlist,arn_dqi_fitdata,color='g',label="$0.98 D^{-0.71}$")
ax1.plot(arn_qaoa_Dlist,arn_qaoadata,color='purple',linestyle="dashed",label="QAOA, p=14, up to O(1/D) corrections")
plt.title("k=3, n=20,000")
ax1.legend()
plt.xlabel("D  (log scale)")
plt.ylabel("$\phi$ $-$ $1/2$  (log scale)")
plt.xscale("log")
plt.yscale("log")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.set_yticks([0.03,0.1,0.4])
ax1.set_yticklabels(["$3 \\times 10^{-2}$","$10^{-1}$","$4 \\times 10^{-1}$"])
ax1.set_xticks([4,10,100,600])
ax1.set_xticklabels(["4","10","100","600"])
fig1.savefig("fig6_constn_new.pdf",bbox_inches='tight')