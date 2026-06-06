#include "xorsat.hpp"
#include "bitmatrix.hpp"
#include "utils.hpp"
#include <cmath>
#include <cassert>


void bruteforce(instance & I) {
    std::vector<size_t> buckets(I.num_cons+1, 0);
    bitvector x;
    x.zeros(I.num_vars);
    for (size_t x_num = 0; x_num < (1u<<I.num_vars); ++x_num) {
        int num_sat = (I.num_cons + 2*evaluate(I, x))/2;
        ++buckets[num_sat];
        assert(x.to_num() == x_num);
        // std::cout << x.to_str() << " sat " << num_sat << "\n";
        x.increment();
    }
    std::cout << "num_x,num_sat\n";
    for (size_t num_sat = 0; num_sat <= I.num_cons; ++num_sat) {
        // std::cout << buckets[num_sat] << " satisfying " << num_sat << " / " << I.num_cons << " clauses\n";
        std::cout << buckets[num_sat] << "," << num_sat << "\n";
    }
}

int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: brute instance.tsv" << std::endl;
        return 0;
    }
    instance I;
    std::string filename = argv[1];
    if(!load_instance(filename, I)) return 0;
    bruteforce(I);
    return 0;
}
