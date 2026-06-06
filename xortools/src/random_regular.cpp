#include "utils.hpp"
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>

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
void random_regular(size_t k, size_t D, size_t bsize, size_t hsize, int planted_value, std::string filename, std::mt19937_64 &eng) {
    std::vector<int> zerow(D, 0);
    std::vector<std::vector<int> >M(k*bsize, zerow);
    std::vector<int> perm(D*bsize);
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
    std::vector<bool> planted_x(k*bsize, 0);
    if (planted_value >= 0) {
        std::cout << "planted solution satisfying " << planted_value << " clauses: x = ";
        for (size_t i=0; i<k*bsize; ++i) {
            planted_x[i] = uniform(eng)%2;
            std::cout << planted_x[i];
        }
        std::cout << "\n";
    }

    outfile << k*bsize << "\t" << D*bsize << std::endl;
    for(size_t con_index = 0; con_index < D*bsize; con_index++) {
        bool target_bit = false;
        if (planted_value >= 0) {
            // Compute onto the target bit the value we'd get from the planted solution
            for (size_t j = 0; j < k; j++) {
                target_bit ^= planted_x[MT[con_index][j]];
            }
            if (planted_value > 0) {
                // Decrement the planted value so we only give planted_x the specified value
                --planted_value;
            } else {
                // Set the target to the opposite of the planted_x result so that it only has
                // the specified value
                target_bit ^= true;
            }
        } else {
            target_bit = (uniform(eng)%2);
        }

        if(con_index < hsize) {
            if(target_bit) outfile << "-137.04\t";
            else outfile << "137.04\t";
        }
        else {
            if(target_bit) outfile << "-0.5000\t";
            else outfile << "0.5000\t";
        }
        for(size_t j = 0; j < k; j++) {
            outfile << MT[con_index][j];
            if(j == k-1) outfile << std::endl;
            else outfile << "\t";
        }
    }
    outfile.close();
}

int main(int argc, char *argv[]) {
    int k,D,bsize;
    int hsize=0;
    int planted_value = -1;
    if (argc == 4 or argc == 6) {
        k = std::stoi(argv[1]);
        D = std::stoi(argv[2]);
        bsize = std::stoi(argv[3]);
        if(k < 2) {
            std::cout << "k must be at least 2." << std::endl;
            return 0;
        }
        if(D <= k) {
            std::cout << "D must exceed k." << std::endl;
            return 0;
        }
        if (argc >= 5) {
            if (argc != 6) {
                std::cout << "must provide both hsize and planted_value" << std::endl;
                return 0;
            }
            hsize = std::stoi(argv[4]);
            if(hsize > D*bsize) {
                std::cout << "Number of hard constraints (hsize) cannot exceed total number of constraints (D*bsize)." << std::endl;
                return 0;
            }
            planted_value = std::stoi(argv[5]);
            if (planted_value > bsize*D) {
                std::cout << "planted_value must be at most bsize*D" << std::endl;
                return 0;
            }
        }
    } else {
        std::cout << "Usage: random_regular k D bsize <hsize> <planted_value> -- note, must provide hsize AND planted value or neither" << std::endl;
        return 0;
    }
    std::random_device rd;
    uint64_t seed = rd();
    std::mt19937_64 eng(seed);
    std::cout << "seed = " << seed << std::endl;
    std::string filename = "instance_" + std::to_string(k) + "_" + std::to_string(D) + "_" + std::to_string(bsize);
    if (hsize > 0) {
        filename += "_h" + std::to_string(hsize);
    }
    if (planted_value >= 0) {
        filename += "_plv" + std::to_string(planted_value);
    }
    filename += ".tsv";
    random_regular(k,D,bsize,hsize,planted_value,filename,eng);
    return 0;
}
