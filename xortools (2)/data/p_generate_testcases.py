#Galois version 0.3.5 doesn't work for me. So, I instead do: pip install galois==0.3.4

import numpy as np
import random
import galois
import io

p = 11
GF = galois.GF(p)
random.seed(1)

def randmat(n):
    array = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(random.randrange(p))
        array.append(row)
    return GF(array)

def randmat_rect(n,m):
    array = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(random.randrange(p))
        array.append(row)
    return GF(array)

def GFp_string(x):
    output = io.StringIO()
    print(x, end="", file=output)
    return output.getvalue()

def save_dense(matrix, filename):
    fo = open(filename, "w")
    rows,cols = matrix.shape
    for row in range(rows):
        for col in range(cols):
            fo.write(GFp_string(matrix[row,col]))
            if col < cols-1:
                fo.write(",")
        fo.write("\n")
    fo.close()

sizes = [4,5,5,5,75,75,75]

A_arr = []
for s in sizes:
    A_arr.append(randmat(s))

D_arr = []
for A in A_arr:
    det = np.linalg.det(A)
    print(det)
    D_arr.append(det)

testnum = 0
for A in A_arr:
    filename = "p_inv_test" + str(testnum) + ".txt"
    save_dense(A, filename)
    if(D_arr[testnum] != 0):
        inversename = "p_inverse" + str(testnum) + ".txt"
        save_dense(np.linalg.inv(A), inversename)
    testnum = testnum + 1

def gen_half_rank(rows, cols):
    half = round(rows/2)
    R = randmat_rect(half, cols)
    for k in range(half):
        i = 0
        j = 0
        while i == j:
            i = random.randrange(half)
            j = random.randrange(half)
        m1 = random.randrange(p-1)+1
        m2 = random.randrange(p-1)+1
        newrow = GF(m1) * R[i:i+1,:] + GF(m2) * R[j:j+1,:]
        R = np.vstack((R,newrow))
    n_rows,n_cols = R.shape
    perm = np.random.permutation(n_rows)
    RP = R[perm[0]:perm[0]+1,:]
    for j in range(1,n_rows):
        RP = np.vstack((RP,R[perm[j]:perm[j]+1,:]))
    print(np.linalg.matrix_rank(RP))
    return RP

B0 = gen_half_rank(4,5)
B1 = gen_half_rank(6,10)
B2 = gen_half_rank(70,60)
B3 = gen_half_rank(30,20)
B4 = gen_half_rank(130,131)

save_dense(B0, "p_testB0.txt")
save_dense(B1, "p_testB1.txt")
save_dense(B2, "p_testB2.txt")
save_dense(B3, "p_testB3.txt")
save_dense(B4, "p_testB4.txt")

Csizes = [1,1,2,3,10,100,201]

C_arr = []
for s in Csizes:
    C_arr.append(randmat(s))

testnum = 0
for C in C_arr:
    filename = "p_testC" + str(testnum) + ".txt"
    save_dense(C, filename)
    print(np.linalg.det(C))
    testnum = testnum + 1

P0A = randmat(2)
P0B = randmat(2)
P0C = np.matmul(P0A,P0B)
save_dense(P0A, "p_testP0A.txt")
save_dense(P0B, "p_testP0B.txt")
save_dense(P0C, "p_testP0C.txt")

P1A = randmat_rect(3,5)
P1B = randmat_rect(5,4)
P1C = np.matmul(P1A,P1B)
save_dense(P0A, "p_testP1A.txt")
save_dense(P0B, "p_testP1B.txt")
save_dense(P0C, "p_testP1C.txt")

P2A = randmat_rect(75,80)
P2B = randmat_rect(80,90)
P2C = np.matmul(P2A,P2B)
save_dense(P0A, "p_testP2A.txt")
save_dense(P0B, "p_testP2B.txt")
save_dense(P0C, "p_testP2C.txt")

P3A = randmat_rect(10,11)
P3B = randmat_rect(11,9)
P3C = np.matmul(P3A,P3B)
save_dense(P0A, "p_testP3A.txt")
save_dense(P0B, "p_testP3B.txt")
save_dense(P0C, "p_testP3C.txt")