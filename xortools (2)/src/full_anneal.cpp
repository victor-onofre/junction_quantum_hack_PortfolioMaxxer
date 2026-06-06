#include <iostream>
#include <random>
#include "xoropt.hpp"

int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: full_anneal instance.tsv" << std::endl;
        return 0;
    }
    xorsat_instance I;
    if(!I.load(argv[1])) return 0;
    std::random_device rd;
    uint64_t seed = rd();
    std::cout << "seed: " << seed << std::endl;
    std::mt19937_64 eng(seed);
    walker w(I);
    w.randomize(eng);
    anneal(w, 5000, 0.0, 3.0, eng);
    return 0;
}