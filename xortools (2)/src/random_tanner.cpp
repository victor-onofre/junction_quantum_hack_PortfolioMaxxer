#include "utils.hpp"
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>

/* Generates tanner code instances.
*/

//Terminal command: g++ -std=c++17 random_tanner.cpp -o test_random_tanner

void random_regular(size_t k, size_t D, size_t n, size_t tan, std::string filename, std::mt19937_64 &eng) {
    std::uniform_int_distribution<unsigned int> uniform;
    size_t m = D * n / k;
    std::vector<bool> false_init(m, false);
    std::vector<std::vector<bool> > adjacency(n * tan, false_init);
    std::vector<size_t> terms(m);
    for(size_t x = 0; x < m; x++) terms[x] = x;
    for(size_t i = 0; i < n; i++) {
        std::shuffle(terms.begin(), terms.end(), eng);
        std::vector<size_t> T(D);
        for(size_t j = 0; j < D; j++) {
            T[j] = terms[j];
        }
        std::sort(T.begin(), T.end());
        for(size_t t = 0; t < tan; t++) {
            for(size_t j = 0; j < D; j++) {
                bool coin = false;
                coin = (uniform(eng)%2);
                if(coin) {
                    adjacency[i * tan + t][T[j]] = true;
                }
            }
        }
    }
    
    //file output
    std::ofstream outfile;
    outfile.open(filename);
    if(!outfile.is_open()) {
        std::cout << "Error: unable to create " << std::endl;
        return;
    }
    outfile << n * tan << "\t" << m << std::endl;
    for(size_t j = 0; j < m; j++) {
        bool target_bit = false;
        target_bit = (uniform(eng)%2);
        if(target_bit) outfile << "-0.5000\t";
        else outfile << "0.5000\t";
        for(size_t i = 0; i < n * tan; i++) {
            if(adjacency[i][j]) {
                outfile << i;
                if(i != n * tan - 1) outfile << "\t";
            }
            if(i == n * tan - 1) outfile << std::endl;
        }
    }
    outfile.close();
}

int main(int argc, char *argv[]) {
    int k, D, n;
    int tan = 1;
    if(argc == 4 or argc == 5) {
        k = std::stoi(argv[1]);
        D = std::stoi(argv[2]);
        n = std::stoi(argv[3]);
        if(argc == 5) {
            tan = std::stoi(argv[4]);
        }
        if(k < 2) {
            std::cout << "k must be at least 2." << std::endl;
            return 0;
        }
        if(D <= k * tan) {
            std::cout << "D must exceed k * tan." << std::endl;
            return 0;
        }
    } else {
        std::cout << "Usage: random_regular k D n <tan>" << std::endl;
        return 0;
    }
    std::random_device rd;
    uint64_t seed = rd();
    std::mt19937_64 eng(seed);
    std::cout << "seed = " << seed << std::endl;
    std::string filename = "instance_tan_" + std::to_string(k) + "_" + std::to_string(D) + "_" + std::to_string(n);
    filename += ".tsv";
    random_regular(k, D, n, tan, filename, eng);
    return 0;
}
