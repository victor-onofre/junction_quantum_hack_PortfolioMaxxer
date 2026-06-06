#ifndef RANDREG_HPP
#define RANDREG_HPP

#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include "modpinverse.hpp"
#include "utils.hpp"

/* This is the parity check matrix from the ensemble of LDPC codes defined in:
  V.V. Zyablov and M. S. Pinsker, Estimation of the error-correction capacity
  for Gallager low-density codes. Problemy Peredachi Informatsii, 11(1):23-26, 1975.
  My notation is not the same as theirs. The dictionary is:
  k = l: number of variables per max-k-SAT clause
  D = h: number of clauses in which each variable appers
  m = hn: number of clauses
  n = ln: number of variables
  Necessarily: nD = mk.
  It is necessary that D > k because otherwise the problem is underconstrained and
  with high probability all clauses can be satisfied.
*/
void random_regular(size_t k, size_t D, size_t bsize, std::string filename, int modulus, std::mt19937_64 &eng, size_t listsize = 1) {
    std::vector<int> zerow(D, 0);
    std::vector<std::vector<int> >M(k*bsize, zerow);
    std::vector<int> perm(D*bsize);
    std::uniform_int_distribution<uint16_t> all_vals(0,modulus-1);
    std::uniform_int_distribution<uint16_t> nonzero_vals(1,modulus-1);
    std::vector<uint16_t>rhs_list;
    for(size_t i = 0; i < D*bsize; i++) perm[i] = i;
    for(size_t a = 0; a < k; a++) {
        std::shuffle(perm.begin(), perm.end(), eng);
        for(size_t b = 0; b < bsize; b++) {
            size_t row_index = a*bsize+b;
            for(size_t entry_index = 0; entry_index < D; entry_index++) {
                M[row_index][entry_index] = perm[bsize*entry_index + row_index%bsize];
            }
        }
    }
    //transpose of sparse matrix
    std::vector<std::vector<int> >MT(D*bsize);
    for(size_t var_index = 0; var_index < k*bsize; var_index++) {
        for(size_t j = 0; j < D; j++) {
            size_t con_index = M[var_index][j];
            MT[con_index].push_back(var_index);
        }
    }
    //file output
    std::ofstream outfile;
    outfile.open(filename);
    if(!outfile.is_open()) {
        std::cout << "Error: unable to create " << std::endl;
        return;
    }
    std::uniform_int_distribution<unsigned int> uniform;
    outfile << k*bsize << "," << D*bsize << "," << modulus << std::endl;
    for(size_t con_index = 0; con_index < D*bsize; con_index++) {
        rhs_list = random_subset<uint16_t>(modulus, listsize, eng);
        for(size_t j = 0; j < listsize; j++) outfile << rhs_list[j] << ","; //the rhs list
        for(size_t j = 0; j < k; j++) {
            outfile << MT[con_index][j] << ":" << nonzero_vals(eng);
            if(j == k-1) outfile << std::endl;
            else outfile << ",";
        }
    }
    outfile.close();
}

bool args_ok(int k, int D, int bsize, int modulus, int listsize) {
    if(k < 2) {
        std::cout << "k must be at least 2." << std::endl;
        return false;
    }
    if(D <= k) {
        std::cout << "D must exceed k." << std::endl;
        return false;
    }
    if(bsize < 1 || bsize > 1E5) {
        std::cout << "Invalid bsize." << std::endl;
        return false;
    }
    if(modulus <= 0 || modulus >= PMAX) {
        std::cout << "Modulus must be a prime smaller than " << PMAX << std::endl;
        return false;
    }
    if(!p_prime[modulus]) {
        std::cout << "Proposed modulus (" << modulus << ") is not prime." << std::endl;
        return false;
    }
    if(listsize < 1 || listsize >= modulus) {
        std::cout << "List size must be at least one and less than modulus." << std::endl;
        return false;
    }
    return true;
}

#endif