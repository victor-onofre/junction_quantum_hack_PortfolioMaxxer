#include <iostream>
#include "xorsat.hpp"
#include "bitmatrix.hpp"
#include "ldpc.hpp"

int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: gaussheur instance.tsv" << std::endl;
        return 0;
    }
    instance I;
    if(!load_instance(argv[1], I)) return 0;
    priority_order(I);
    bitmatrix H;
    H = parity_check_matrix(I);
    bitvector t = target(I);
    bitvector x;
    int satisfied = gaussheur(H, t, x); 
    std::cout << "H has rank: " << H.rank() << std::endl;
    std::cout << satisfied << " clauses satisfied out of " << H.num_cols() << std::endl;
    std::cout << "objective value: " << evaluate(I, x) << std::endl;
    std::cout << "solution: " << std::endl;
    std::cout << x << std::endl;
    return 0;
}
