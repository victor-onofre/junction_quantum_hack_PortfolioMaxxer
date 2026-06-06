#include <iostream>
#include <cstdint>
#include <cmath>
#include "utils.hpp"
#include "maxlinsat.hpp"

bool is_prime(uint64_t x) {
    double sqrtx = sqrt((double)x);
    int stop = ceil(sqrtx);
    for(int i = 2; i <= stop; i++) if((x % i) == 0) return false;
    return true;
}

bool is_primitive(uint64_t x, uint64_t p) {
    if(x == 0) return false;
    uint64_t val = x % p;
    for(uint64_t power = 1; power < p-1; power++) {
        if(val == 1) return false;
        val = (val * x) % p;
    }
    return true;
}

uint64_t prime_atleast(uint64_t x) {
    uint64_t i;
    //for any x > 1 there's always a prime between x and 2x
    for(i = x; i <= 2*x; i++) {
        if(is_prime(i)) return i;
    }
    std::cout << "This can't happen." << std::endl;
    return i;
}

uint64_t prime_atmost(uint64_t x) {
    uint64_t i;
    //for any x > 1 there's always a prime between x and 2x
    for(i = x; i >= x/2; i--) {
        if(is_prime(i)) return i;
    }
    std::cout << "This can't happen." << std::endl;
    return i;
}

//assumes p is prime
uint64_t smallest_primitive(uint64_t p) {
    uint64_t x = 2;
    while(!is_primitive(x,p)) x++;
    return x;
}

int main(int argc, char *argv[]) {
    if(argc != 3) {
        std::cout << "Usage: rsinstance n m" << std::endl;
        return 0; 
    }
    int n = std::stoi(argv[1]);
    int m = std::stoi(argv[2]);
    if(m <= n) {
        std::cout << "Error: m must be bigger than n." << std::endl;
        return 0;
    }
    if(n < 2 || m > 1E7) {
        std::cout << "Error: parameters out of range." << std::endl;
        return 0;
    }
    //Initialize random number generator for making the lists--------------------------
    std::random_device rd;
    uint64_t seed = rd();
    std::cout << "seed: " << seed << std::endl;
    std::mt19937_64 eng(seed);
    //Bump up m to prime_atleast(m) and bump up n as needed to preserve m/n------------
    uint64_t p = prime_atleast(m);
    double ratio = (double)p/(double)m;
    double newn = ratio*(double)n;
    n = std::round(newn);
    m = p;
    //----------------------------------------------------------------------------------
    uint64_t gamma = smallest_primitive(p);
    modinstance I;
    I.num_vars = n;
    I.num_cons = m;
    I.modulus = p;
    std::vector<std::vector<uint64_t>> v(I.num_vars, std::vector<uint64_t>(I.num_cons,0));
    std::vector<std::vector<bool>> f(I.num_cons, std::vector<bool>(I.modulus, false));
    int listsize = std::floor(0.5*(double)p);
    uint64_t base = gamma;
    for(size_t i = 0; i < I.num_vars; i++) {
        uint64_t val = 1;
        for(size_t j = 0; j < I.num_cons; j++) {
            v[i][j] = val; //v[i][j] = gamma^{i times j}
            val = (val * base) % p;
        }
        base = (base * gamma) % p;
    }
    for(size_t j = 0; j < I.num_cons; j++) {
        std::vector <size_t> list = random_subset<size_t>(p, listsize, eng);
        for(size_t v : list) f[j][v] = true;
    }
    I.v = v;
    I.f = f;
    std::string filename = "rs" + std::to_string(n) + "_" + std::to_string(m) + ".csv";
    save_modinstance(filename, I);
    std::cout << "Output written to " << filename << std::endl;
    return 0;
}