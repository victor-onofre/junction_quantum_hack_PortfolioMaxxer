#include <iostream>
#include <random>
#include "xoropt.hpp"

//Sergei Isakov's parameters:
int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: partial_anneal instance.tsv" << std::endl;
        return 0;
    }
    xorsat_instance I;
    if(!I.load(argv[1])) return 0;
    std::random_device rd;
    uint64_t seed = rd();
    std::cout << "seed: " << seed << std::endl;
    bitvector x, best_x;
    int best_val = I.num_cons();
    for(size_t rep = 0; rep < 200; rep++) {
        std::cout << "Rep: " << rep << std::endl;
        int val = partial_anneal(I, 21, x, seed+rep);
        if(val < best_val) {
            best_val = val;
            best_x = x;
        }
    }
    std::cout << "Best val: " << best_val << std::endl;
    std::cout << I.num_cons() - best_val << " clauses satisfied out of " << I.num_cons() << std::endl;
    std::cout << "Best x:\n" << best_x << std::endl;
    return 0;
}