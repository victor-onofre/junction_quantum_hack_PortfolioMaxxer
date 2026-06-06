import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

f = open('wide_summary.txt', 'r')
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

n = 31216
m = 50000

sat = []
sweeps = []
updates = []
satfrac = []
for linetokens in tokenarray:
    namestring = linetokens[0]
    namestring = namestring.replace("_"," ")
    namestring = namestring.replace("."," ")
    nametokens = namestring.split()
    sweepcount = int(nametokens[1])
    satcount = int(linetokens[7])
    sweeps.append(sweepcount)
    sat.append(satcount)
    satfrac.append(satcount/m)
    updates.append(sweepcount*n)


# Convert to NumPy arrays
x = np.array(updates, dtype=float)
y = np.array(satfrac, dtype=float)

#fit power law model---------------------------------------------------------------------------

def model(xval, aval, bval, cval):
    return aval - bval * pow(xval,-cval)

# Initial guesses for a, b, c
p0 = [0.9, 1.0E-11, 0.5]

# Correct bounds: two sequences of length 3
lb = [-np.inf, -np.inf,  0.0]
ub = [ np.inf,  np.inf,  np.inf]

popt, pcov = curve_fit(
    model, x, y, p0=p0,
    bounds=(lb, ub)
)
a_fit, b_fit, c_fit = popt
print(f"Power Law Fitted parameters: a = {a_fit:.6f}, b = {b_fit:.6f}, c = {c_fit:.6f}")

labelstring = f"Fit: ${a_fit:.2f} - {b_fit:.2f} \\times N^{{-{c_fit:.2f}}}$"

#prepare a smooth curve for plotting the power-law fit but this
#time make the points spaced evenly on a log scale
xlog = np.linspace(np.log(x.min()), np.log(x.max()),5000)
x_lfit = np.exp(xlog)
y_lfit = model(x_lfit, a_fit, b_fit, c_fit)

# fit sqrt model function-----------------------------------------------------------------------------

def rectify(xval):
    # for each element: if xval_i <= 1.0, replace with 1.0, else keep xval_i
    return np.where(xval <= 1.0, 1.0, xval)

def smodel(xval, aval, bval, cval):
    # ensure the argument of log and sqrt stays positive
    return cval + aval * np.sqrt(np.log(rectify(bval * xval)))

# Initial guesses for a, b, c
p0 = [0.07, 0.002, 0.5]

# Correct bounds: two sequences of length 3
lb = [-np.inf,    0.0,  -np.inf]
ub = [ np.inf,  np.inf,  np.inf]

popt, pcov = curve_fit(
    smodel, x, y, p0=p0,
    bounds=(lb, ub)
)
as_fit, bs_fit, cs_fit = popt

print(f"Sqrt Fitted parameters: a = {as_fit:.6f}, b = {bs_fit:.6f}, c = {cs_fit:.6f}")

#slabelstring = (
#    f"Fit: ${cs_fit:.2f} + {as_fit:.2f} \\times "
#    f"\\sqrt{{\\log({bs_fit:.2e}\\,N)}}$"
#)

slabelstring = "Fit: $0.64 + 0.049 \\times \\sqrt{\\log (N \\times 1.18 \\times 10^{-5})}$"

#prepare a smooth curve for plotting the sqrt fit but this
#time make the points spaced evenly on a log scale
xlog = np.linspace(np.log(x.min()), np.log(x.max()),5000)

xslf = []
yslf = []
for xval in xlog:
    if bs_fit*np.exp(xval) >= 1.0:
        xslf.append(np.exp(xval))
        yslf.append(smodel(np.exp(xval), as_fit, bs_fit, cs_fit))

sxl_fit = np.array(xslf, dtype=float)
syl_fit = np.array(yslf, dtype=float)

# --- Second plot: log‐scale x‐axis ---
fig, ax = plt.subplots(figsize=(8, 5))
plt.scatter(x, y, color='b', marker='.', label="Empirical data")
plt.plot(x_lfit, y_lfit, 'r-', linewidth=2, label=labelstring)
plt.plot(sxl_fit, syl_fit, 'g--', linewidth=1, label=slabelstring)
plt.xscale('log')                          # ← switch to log‐scale here
plt.xlabel("Metropolis Updates (log scale)")
plt.ylabel("Fraction Satisfied")
plt.title("Convergence of Simulated Annealing on Tuned Irregular Instance")
plt.legend()
plt.grid(True)      # show grid on minor ticks too
plt.tight_layout()
fig.savefig("irregular_fits.pdf")