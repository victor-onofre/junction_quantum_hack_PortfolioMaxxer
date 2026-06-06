#include "panneal.hpp"

int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: salist instance.csv" << std::endl;
        return 0;
    }
    lsinstance I;
    std::random_device rd;
    uint64_t seed = rd();
    std::cout << "seed: " << seed << std::endl;
    std::mt19937_64 eng(seed);
    std::string filename = argv[1];
    if(!load_lsinstance(filename, I)) return 0;
    anneal<lsinstance>(I, eng);
    return 0;
}
