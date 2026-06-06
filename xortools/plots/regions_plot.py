import numpy as np
from scipy.optimize import bisect
import matplotlib
matplotlib.use('Agg')    # before importing pyplot
import matplotlib.pyplot as plt

# Enable LaTeX rendering for text elements
plt.rcParams.update({
    "text.usetex": True,         # Use LaTeX for text rendering
    "font.family": "serif",     # Use a serif font family
    "font.serif": ["Computer Modern Roman"],   # Specify Computer Modern as the serif font
    "text.latex.preamble": r'\usepackage{amsmath} \usepackage{amssymb}', # Use some helpful packages
})

# Customize further if needed (e.g., font size, figure size)
plt.rcParams.update({
    "font.size": 11,             # Set the font size (adjust to your preference)
    "figure.figsize": (6, 4),     # Set the default figure size (adjust as needed)
})

def load_and_parse(filename):
    try:
        f = open(filename, 'r')
    except FileNotFoundError:
        print("File not found")
        return []
    tokenarray=[]
    line = f.readline()
    while len(line) > 0:
        if line[0] != '#':
            line = line.replace("," , " ") #handle both tsv and csv
            line = line.replace(":" , " ")
            line = line.replace("(", "")
            line = line.replace(")", "")
            tokens = line.split()
            numread = len(tokens)
            tokenarray.append(tokens)
        line = f.readline()
    f.close()
    dimensions = (40, 120, 5)
    s_data = np.full(dimensions, -1)
    m_data = np.full(dimensions, -1)
    for line in tokenarray:
        sat = int(line[7])
        m = int(line[10])
        namestring = line[0]
        namestring = namestring.replace("_"," ")
        namestring = namestring.replace("."," ")
        nametokens = namestring.split()
        k = int(nametokens[1])
        D = int(nametokens[2])
        trial = int(nametokens[3])
        if k < 3 or k >= 40:
            print("Bad k")
            continue
        if D < k+1 or D >= 120:
            print("Bad D")
            continue
        if trial < 0 or trial >= 5:
            print("Bad trial")
            continue
        s_data[k][D][trial] = sat
        m_data[k][D][trial] = m
    return s_data, m_data

sdat, mdat = load_and_parse("lx_summary.txt")

#returns 1 if all entries of t = val, 0 if no entries of t = val, 2 if some = val and some don't
def tuplecat(t,val):
    allv = True
    nonev = True
    for v in t:
        if v != val:
            allv = False
        if v == val:
            nonev = True
    if allv:
        return 1
    if nonev:
        return 0
    return 2

def all_equal(t):
    val = t[0]
    for index in range(1,len(t)):
        if t[index] != val:
            return False
    return True

m_list = []
k_list = []
D_list = []
maxs_list = []
for k in range(0,40):
    for D in range(0,120):
        mtuple = mdat[k][D]
        stuple = sdat[k][D]
        mcat = tuplecat(mtuple, -1)
        scat = tuplecat(stuple, -1)
        if mcat == 2 or scat == 2 or scat != mcat or not all_equal(mtuple):
            print("Bad tuple")
            continue
        if mcat == 0: #which implies scat == 0
            maxs_list.append(max(stuple))
            m_list.append(mtuple[0])
            k_list.append(k)
            D_list.append(D)

frac_list = []
for index in range(0,len(maxs_list)):
    frac_list.append(maxs_list[index]/m_list[index])

def binary_entropy(p):
    # Handle edge cases carefully
    p = np.clip(p, 1e-12, 1-1e-12)
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def inverse_binary_entropy(h, side='lower'):
    #Numerically invert binary entropy.
    # h : Entropy value between 0 and 1.
    # side : 'lower' or 'upper' Which solution to return (p <= 0.5 or p >= 0.5).
    if not (0 <= h <= 1):
        raise ValueError("Entropy value must be between 0 and 1.")

    if side == 'lower':
        # Solve in [0, 0.5]
        return bisect(lambda p: binary_entropy(p) - h, 0.0, 0.5)
    elif side == 'upper':
        # Solve in [0.5, 1]
        return bisect(lambda p: binary_entropy(p) - h, 0.5, 1.0)
    else:
        raise ValueError("side must be 'lower' or 'upper'")

def test_inverse_binary_entropy():
    np.random.seed(42)  # For reproducibility
    errors = []
    for _ in range(1000):
        p = np.random.uniform(0, 0.5)
        h = binary_entropy(p)
        p_hat = inverse_binary_entropy(h, side='lower')
        error = abs(p_hat - p)
        errors.append(error)
        assert error < 1e-6, f"Test failed: p={p}, p_hat={p_hat}, error={error}"
    print(f"All 1000 tests passed. Max error: {max(errors):.2e}")

test_inverse_binary_entropy()

# Holevo chi for the channel: |0> -> sqrt(p) |0> + sqrt(1-p) |1> and |1> -> sqrt(p) |1> + sqrt(1-p) |0>
def chi(p):
    return binary_entropy(0.5-np.sqrt(p*(1.0-p)))

def inverse_chi(c):
    if not (0 <= c <= 1):
        raise ValueError("Chi value must be between 0 and 1.")
    # Solve in [0, 0.5]
    return bisect(lambda p: chi(p) - c, 0.0, 0.5)

def test_inverse_chi():
    np.random.seed(42)  # For reproducibility
    errors = []
    for _ in range(1000):
        c = np.random.uniform(0, 0.5)
        x = chi(c)
        c_hat = inverse_chi(x)
        error = abs(c_hat - c)
        errors.append(error)
        assert error < 1e-6, f"Test failed: p={p}, p_hat={p_hat}, error={error}"
    print(f"All 1000 tests passed. Max error: {max(errors):.2e}")

test_inverse_chi()

# the satisfaction fraction achieved by DQI with Holevo-bound decoding
def q_fdqi(k, D):
    p = inverse_chi(1-k/D)
    return 1/2+np.sqrt(p*(1-p))

# the satisfaction fraction achieved by DQI with Shannon-bound decoding
def fdqi(k, D):
    p = inverse_binary_entropy(k/D)
    return 1/2+np.sqrt(p*(1-p))

# the satisfaction fraction achieved by Prange's algorithm
def fprange(k,D):
    return 0.5+0.5*k/D

DQIkwins = []
DQIDwins = []
SAkwins = []
SADwins = []
Prangekwins = []
PrangeDwins = []
for index in range(0,len(frac_list)):
    k = k_list[index]
    D = D_list[index]
    fSA = frac_list[index]
    fP = fprange(k,D)
    fDQI = fdqi(k,D)
    arr = np.array([fDQI, fP, fSA])
    i = arr.argmax()
    if i == 0:
        DQIkwins.append(k)
        DQIDwins.append(D)
    if i == 1:
        Prangekwins.append(k)
        PrangeDwins.append(D)
    if i == 2:
        SAkwins.append(k)
        SADwins.append(D)

#Make the Shannon regions plot:---------------------------------------------------

red = np.array([196, 30, 58, 255])/255
yellow = np.array([255, 191, 0, 255])/255
blue = np.array([64,143,239, 255])/255
green = np.array([50, 205, 50, 255])/255
purple = np.array([93, 63, 211, 255])/255
black = np.array([0,0,0, 255])
colors = [yellow, blue, red, blue, green, purple]

# Determine canvas size
kmin = min(np.min(DQIkwins), np.min(SAkwins))
kmax = max(np.max(DQIkwins), np.max(SAkwins))
Dmin = min(np.min(DQIDwins), np.min(SADwins))
Dmax = max(np.max(DQIDwins), np.max(SADwins))

height = Dmax - Dmin + 1
width = kmax - kmin + 1
img = np.zeros((height, width, 4))  # RGBA

# Assign colors (fully opaque)
dqicolor = blue
sacolor = (yellow + red)/2
prangecolor = green

for k, D in zip(DQIkwins, DQIDwins):
    img[D - Dmin, k - kmin] = dqicolor

for k, D in zip(SAkwins, SADwins):
    img[D - Dmin, k - kmin] = sacolor

for k, D in zip(Prangekwins, PrangeDwins):
    img[D - Dmin, k - kmin] = prangecolor

# Plot
fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.imshow(img, extent=[kmin, kmax, Dmin, Dmax], interpolation="nearest", aspect='auto', origin='lower')
ax1.set_xlabel("$k$")
ax1.set_ylabel("$D$")

# Manually create legend
from matplotlib.patches import Patch
ax1.legend(handles=[
    Patch(color=dqicolor, label="DQI Shannon Bound"),
    Patch(color=sacolor, label="Simulated Annealing"),
    Patch(color=prangecolor, label="Prange's Algorithm")
])

fig1.savefig('shannon_regions.pdf', bbox_inches="tight")

#Make the Holevo regions plot (which is boring because DQI always wins):----------------------------

q_DQIkwins = []
q_DQIDwins = []
q_SAkwins = []
q_SADwins = []
q_Prangekwins = []
q_PrangeDwins = []
for index in range(0,len(frac_list)):
    k = k_list[index]
    D = D_list[index]
    fSA = frac_list[index]
    fP = fprange(k,D)
    fDQI = q_fdqi(k,D)
    arr = np.array([fDQI, fP, fSA])
    i = arr.argmax()
    if i == 0:
        q_DQIkwins.append(k)
        q_DQIDwins.append(D)
    if i == 1:
        q_Prangekwins.append(k)
        q_PrangeDwins.append(D)
    if i == 2:
        q_SAkwins.append(k)
        q_SADwins.append(D)

red = np.array([196, 30, 58, 255])/255
yellow = np.array([255, 191, 0, 255])/255
blue = np.array([64,143,239, 255])/255
green = np.array([50, 205, 50, 255])/255
purple = np.array([93, 63, 211, 255])/255
black = np.array([0,0,0, 255])
colors = [yellow, blue, red, blue, green, purple]

# Determine canvas size
kmin = np.min(q_DQIkwins)
kmax = np.max(q_DQIkwins)
Dmin = np.min(q_DQIDwins)
Dmax = np.max(q_DQIDwins)

height = Dmax - Dmin + 1
width = kmax - kmin + 1
img = np.zeros((height, width, 4))  # RGBA

# Assign colors (fully opaque)
dqicolor = blue
sacolor = (yellow + red)/2
prangecolor = green

for k, D in zip(q_DQIkwins, q_DQIDwins):
    img[D - Dmin, k - kmin] = dqicolor

for k, D in zip(q_SAkwins, q_SADwins):
    img[D - Dmin, k - kmin] = sacolor

for k, D in zip(q_Prangekwins, q_PrangeDwins):
    img[D - Dmin, k - kmin] = prangecolor

# Plot
fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.imshow(img, extent=[kmin, kmax+1, Dmax+1, Dmin], interpolation="nearest", aspect='auto', origin='lower')
ax1.set_xlabel("$k$")
ax1.set_ylabel("$D$")

# Manually create legend
from matplotlib.patches import Patch
ax1.legend(handles=[
    Patch(color=dqicolor, label="DQI Holevo Bound"),
    Patch(color=sacolor, label="Simulated Annealing"),
    Patch(color=prangecolor, label="Prange's Algorithm")
])

fig1.savefig('holevo_regions.pdf', bbox_inches="tight")
