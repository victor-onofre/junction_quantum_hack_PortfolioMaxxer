#!/usr/bin/env python3

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import math
import sys

# Define the matrix construction function
def construct_matrix(l, m):
    size = l + 1  # The matrix is (l+1) by (l+1)

    # Define the diagonals of the tridiagonal matrix
    main_diag = np.zeros(size)
    upper_diag = np.zeros(size - 1)
    lower_diag = np.zeros(size - 1)

    for i in range(size - 1):
        upper_diag[i] = math.sqrt((i+1) * (m-i))
        lower_diag[i] = math.sqrt((m-i) * (i+1))

    # Use scipy's diags to create a sparse matrix
    M = diags([lower_diag, main_diag, upper_diag], offsets=[-1, 0, 1], format='csr')

    return M

# Solver function to compute the largest eigenvalue
def compute_largest_eigenvalue(l, m):
    # Construct the matrix
    M = construct_matrix(l, m)

    # Use scipy.sparse.linalg.eigsh to compute the largest eigenvalue
    # eigsh is specialized for Hermitian/symmetric matrices
    eigenvalues, _ = eigsh(M, k=1, which='LA')  # 'LA' means Largest Algebraic eigenvalue

    # Return the largest eigenvalue
    return eigenvalues[0]

# Main program to take command-line arguments and compute the largest eigenvalue
if __name__ == "__main__":
    # Check if enough arguments are passed
    if len(sys.argv) != 3:
        print("Usage: python program.py <m> <l>")
        sys.exit(1)

    # Read command-line arguments
    m = int(sys.argv[1])  # First argument is m
    l = int(sys.argv[2])  # Second argument is l

    # Compute the largest eigenvalue
    largest_eigenvalue = compute_largest_eigenvalue(l, m)

    # Output the result
    print(str(m) + "\t" + str(l) + "\t" + str(largest_eigenvalue))
