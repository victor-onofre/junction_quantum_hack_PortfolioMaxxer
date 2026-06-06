#include "xorsat.hpp"
#include "bitmatrix.hpp"
#include "utils.hpp"

int main(int argc, char *argv[]) {
    if(argc != 2 && argc != 3) {
        std::cout << "Usage: greedy instance.tsv <solution.sol>" << std::endl;
        return 0;
    }
    instance I;
    std::string filename = argv[1];
    if(!load_instance(filename, I)) return 0;
    bitvector init(I.num_vars);
    std::random_device rd;
    uint64_t seed = rd();
    std::mt19937_64 eng(seed);
    if(argc == 3) {
        if(!init.load_dense(argv[2])) return 0;
    }
    else {
        std::cout << "seed: " << seed << std::endl;
        init.random(eng);
    }
    bitvector sol;
    descend(I, init, sol, true);
    return 0;
}
