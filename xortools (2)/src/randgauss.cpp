#include <iostream>
#include <random>
#include <algorithm>
#include "xorsat.hpp"
#include "bitmatrix.hpp"
#include "ldpc.hpp"

#define NUMTRIALS 100

int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: gaussheur instance.tsv" << std::endl;
        return 0;
    }
    instance I;
    if(!load_instance(argv[1], I)) return 0;
    std::random_device rd;
    uint64_t seed = rd();
    std::cout << "seed: " << seed << std::endl;
    std::mt19937_64 eng(seed);
    priority_order(I);
    bitmatrix H;
    H = parity_check_matrix(I);
    bitvector t = target(I);
    std::cout << "H has rank: " << H.rank() << std::endl;
    if(t.size() != H.num_cols()) {
        std::cout << "I made a mistake in my reasoning." << std::endl;
        return 0;
    }
    std::vector<size_t> permutation(H.num_cols());
    for(size_t i = 0; i < H.num_cols(); i++) permutation[i] = i;
    int maxsat = 0;
    int sat, decsat;
    bitvector bestx;
    bitvector x,decx;
    for(int trial = 0; trial < NUMTRIALS; trial++) {
        sat = gaussheur(H, t, x);
        decsat = descend(I, x, decx, false);
        if(decsat > maxsat) {
            maxsat = decsat;
            bestx = decx;
        }
        std::cout << "trial: " << trial+1 << "/" << NUMTRIALS << " result: " << sat << " -> " << decsat << "/" << I.num_cons << " clauses satisfied." << std::endl;
        //std::cout << "objective value: " << evaluate(I, x) << std::endl; //redundant
        std::shuffle(permutation.begin(), permutation.end(), eng);
        H = permute_columns(H, permutation);
        t = permute_bits(t, permutation);
    }
    std::cout << "Best solution:\n" << bestx << std::endl;
    std::cout << "Maximum satisfaction: " << maxsat << std::endl;
    double obj = 0.5*(double)I.num_cons - (double)maxsat;
    if(evaluate(I, bestx) != obj) {
        std::cout << "Error: " << std::endl;
        std::cout << "Objective value: " << obj << std::endl;
        std::cout << "Objective value by direct evaluation: " << evaluate(I, bestx) << std::endl;
    }
    else std::cout << "Objective value: " << obj << std::endl;
    return 0;
}
