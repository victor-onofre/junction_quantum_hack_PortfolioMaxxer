#include "utils.hpp"
#include "randreg.hpp"
#include <vector>
#include <iostream>

int main(int argc, char *argv[]) {
    int k,D,bsize,modulus,listsize;
    if(argc == 6) {
        k = std::stoi(argv[1]);
        D = std::stoi(argv[2]);
        bsize = std::stoi(argv[3]);
        modulus = std::stoi(argv[4]);
        listsize = std::stoi(argv[5]);
        if(!args_ok(k,D,bsize,modulus,listsize)) return 0;
    }
    else {
        std::cout << "Usage: lrandreg k D bsize modulus listsize" << std::endl;
        return 0;
    }
    std::random_device rd;
    uint64_t seed = rd();
    std::mt19937_64 eng(seed);
    std::cout << "seed = " << seed << std::endl;
    std::string filename = "lx" + std::to_string(k) + "_" + std::to_string(D) + "_" + std::to_string(bsize) + "_" + std::to_string(modulus) + ".csv";
    random_regular(k,D,bsize,filename,modulus,eng,(size_t)listsize);
    return 0;
}
