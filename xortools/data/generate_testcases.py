#Galois version 0.3.5 doesn't work for me. So, I instead do: pip install galois==0.3.4

import numpy as np
import random
import galois

GF = galois.GF(2)

random.seed(1)

def randmat(n):
    array = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(random.randrange(2))
        array.append(row)
    return GF(array)

def randmat_rect(n,m):
    array = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(random.randrange(2))
        array.append(row)
    return GF(array)

def GF2_string(x):
    if x == 0:
        return '0'
    if x == 1:
        return '1'
    
def save_dense(matrix, filename):
    fo = open(filename, "w")
    rows,cols = matrix.shape
    for row in range(rows):
        for col in range(cols):
            fo.write(GF2_string(matrix[row,col]))
        fo.write("\n")
    fo.close()

sizes = [4,5,5,5,75,75,75]

A_arr = []
for s in sizes:
    A_arr.append(randmat(s))

D_arr = []
for A in A_arr:
    D_arr.append(np.linalg.det(A))

testnum = 0
for A in A_arr:
    filename = "inv_test" + str(testnum) + ".txt"
    save_dense(A, filename)
    if(D_arr[testnum] != 0):
        inversename = "inverse" + str(testnum) + ".txt"
        save_dense(np.linalg.inv(A), inversename)
    testnum = testnum + 1

B0 = randmat_rect(4,5)
B1 = randmat_rect(6,10)
B2 = randmat_rect(70,60)
B3 = randmat_rect(30,20)
B4 = randmat_rect(131,131)

save_dense(B0, "testB0.txt")
save_dense(B1, "testB1.txt")
save_dense(B2, "testB2.txt")
save_dense(B3, "testB3.txt")
save_dense(B4, "testB4.txt")

print("rank0 = " + str(np.linalg.matrix_rank(B0)))
print("rank1 = " + str(np.linalg.matrix_rank(B1)))
print("rank2 = " + str(np.linalg.matrix_rank(B2)))
print("rank3 = " + str(np.linalg.matrix_rank(B3)))
print("rank4 = " + str(np.linalg.matrix_rank(B4)))

Csizes = [1,1,2,3,10,100,201]

C_arr = []
for s in Csizes:
    C_arr.append(randmat(s))

testnum = 0
for C in C_arr:
    filename = "testC" + str(testnum) + ".txt"
    save_dense(C, filename)
    print(np.linalg.det(C))
    testnum = testnum + 1

P0A = randmat(2)
P0B = randmat(2)
P0C = np.matmul(P0A,P0B)
save_dense(P0A, "testP0A.txt")
save_dense(P0B, "testP0B.txt")
save_dense(P0C, "testP0C.txt")

P1A = randmat_rect(3,5)
P1B = randmat_rect(5,4)
P1C = np.matmul(P1A,P1B)
save_dense(P0A, "testP1A.txt")
save_dense(P0B, "testP1B.txt")
save_dense(P0C, "testP1C.txt")

P2A = randmat_rect(75,80)
P2B = randmat_rect(80,90)
P2C = np.matmul(P2A,P2B)
save_dense(P0A, "testP2A.txt")
save_dense(P0B, "testP2B.txt")
save_dense(P0C, "testP2C.txt")

P3A = randmat_rect(10,11)
P3B = randmat_rect(11,9)
P3C = np.matmul(P3A,P3B)
save_dense(P0A, "testP3A.txt")
save_dense(P0B, "testP3B.txt")
save_dense(P0C, "testP3C.txt")
