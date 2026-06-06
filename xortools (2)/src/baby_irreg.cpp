#include <iostream>
#include <random>
#include "utils.hpp"
#include "bitmatrix.hpp"
#include "xoropt.hpp"

//Specifies clause-degree distribution and number of variables. Does everything else randomly.
xorsat_instance baby_irreg(size_t n, const std::vector<size_t> &degrees, const std::vector<size_t> &counts, std::mt19937_64 &eng) {
    xorsat_instance I;
    bitmatrix B;
    if(degrees.size() != counts.size()) {
        std::cout << "Error: invalid input to baby_irreg." << std::endl;
        return I;
    }
    bitvector row(n);
    for(size_t index = 0; index < degrees.size(); index++) {
        size_t count = counts[index];
        size_t degree = degrees[index];
        for(size_t rep = 0; rep < count; rep++) {
            std::vector<size_t> subset = random_subset<size_t>(n, degree, eng);
            row.zeros(n);
            for(size_t bit_index : subset) row.set(bit_index, 1);
            B.append_below(row);
        }
    }
    I.BT = B.transpose();
    bitvector v(I.BT.num_cols());
    v.random(eng);
    I.v = v;
    return I;
}

int main() {
    std::random_device rd;
    uint64_t seed = rd();
    std::cout << "seed: " << seed << std::endl;
    std::mt19937_64 eng(seed);
    std::vector<size_t> degrees{5, 20};
    std::vector<size_t> counts{150, 50};
    xorsat_instance I = baby_irreg(100, degrees, counts, eng);
    I.save("baby_irreg1.tsv");
    return 0;
}