#include <iostream>
#include "xorsat.hpp"
#include "bitmatrix.hpp"

int main(int argc, char *argv[]) {
    instance I;
    bitvector x;
    double val;
    if(argc != 3) {
        std::cout << "Usage: evalxor instance.tsv solution.sol" << std::endl;
        return 0;
    }
    if(!load_instance(argv[1], I)) return 0;
    if(!x.load_dense(argv[2])) return 0;
    //Throughout this codebase our sign convention is that the value is the number of unsatisfied clauses minus
    //the number of satisfied clauses. That is, this is a cost function to minimize.
    val = evaluate(I, x);
    std::cout << val << std::endl;
    return 0;
}