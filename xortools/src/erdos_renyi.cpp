#include <iostream>
#include "xorsat.hpp"
#include "ldpc.hpp"
#include "bitmatrix.hpp"

int main(int argc, char *argv[]) {
    if(argc != 4) {
        std::cout << "Usage: erdos_renyi n m p" << std::endl;
        return 0;
    }
    int n,m;
    double p;
    n = std::stoi(argv[1]);
    m = std::stoi(argv[2]);
    p = std::stod(argv[3]);
    if(n <= 0 || m <= n || m > 1E7 || p <= 0.0 || p >= 1.0) {
        std::cout << "Invalid parameters." << std::endl;
        return 0;
    }
    std::random_device rd;
    uint64_t seed = rd();
    std::mt19937_64 eng(seed);
    std::cout << "seed = " << seed << std::endl;
    bitmatrix H;
    H.zeros(n,m);
    std::bernoulli_distribution bd(p);
    for(size_t i = 0; i < H.num_rows(); i++) {
        for(size_t j = 0; j < H.num_cols(); j++) {
            if(bd(eng)) H.set(i,j,1);
        }
    }
    instance I = PCM_to_instance(H);
    std::string filename = "er" + std::to_string(n) + "_" + std::to_string(m) + "_" + argv[3] + ".tsv";
    I.U[0].b.random(eng);
    save_instance(filename, I);
    return 0;
}