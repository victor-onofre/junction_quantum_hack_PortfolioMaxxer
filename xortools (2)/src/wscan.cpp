#include <iostream>
#include <bitset>
#include <bit>
#include <cstdint>
#include <random>
#include <vector>
#include <fstream>
#include "utils.hpp"

std::string binstring(uint64_t x, size_t n) {
    std::bitset<64> allbits;
    allbits = (std::bitset<64>)x;
    std::string returnval(n, '0');
    for(size_t i = 0; i < n; i++) if(allbits[i]) returnval[i] = '1';
    return returnval;
}

bool decent(const std::vector<uint64_t> &b, int m) {
    int count[64];
    for(size_t i = 0; i < 64; i++) count[i] = 0;
    for(size_t j = 0; j < b.size(); j++) {
        for(size_t i = 0; i < m; i++) {
            if(b[j] & (uint64_t)(1 << i)) count[i]++;
        }
    }
    for(size_t i = 0; i < m; i++) if(count[i] <= 1) return false;
    return true;
}

std::vector<uint64_t> gen_mat(int n, int m, int D, std::mt19937_64 &eng) {
    std::vector<uint64_t>b(n);
    int attempt;
    for(attempt = 0; attempt < 100000; attempt++) {
        std::vector <int> all(m);
        for(int j = 0; j < m; j++) all[j] = j;
        for(int i = 0; i < n; i++) {
            b[i] = 0;
            std::vector <int>subset;
            std::sample(all.begin(), all.end(), back_inserter(subset), D, eng);
            for(const int & index : subset) b[i] ^= 1ull<<index;
        }
        if(decent(b,m)) {
            std::cout << "Created a decent generator matrix on attempt " << attempt << std::endl;
            return b;
        }
    }
    std::cout << "Failed to create a decent generator matrix after " << attempt << " attempts" << std::endl;
    return b;
}

std::vector<uint64_t> gen_dense_mat(int n, int m, std::mt19937_64 &eng) {
    std::vector<uint64_t>b(n);
    std::uniform_int_distribution<uint64_t>  distr(0, (1ull<<m)-1ull);
    for(size_t i = 0; i < n; i++) b[i] = distr(eng);
    std::cout << "Created random dense generator matrix" << std::endl;
    return b;
}

int main(int argc, char *argv[]) {
    if(argc != 4 && argc != 5) {
        std::cout << "Usage: wscan n m D <trials>" << std::endl;
        return 0;
    }
    int n = std::atoi(argv[1]);
    int m = std::atoi(argv[2]);
    int D = std::atoi(argv[3]);
    int trials = 1;
    if(argc == 5) trials = std::atoi(argv[4]);
    if(n > m || m < 0 || m > 64 || D > m) {
        std::cout << "Invalid parameters" << std::endl;
        return 0;
    }
    std::random_device rd;
    uint64_t seed = rd();
    std::mt19937_64 eng(seed);
    std::cout << "seed = " << seed << std::endl;
    std::vector<std::vector<uint64_t>> allcounts;
    for(int trial = 0; trial < trials; trial++) {
        std::vector<uint64_t> count(m+1, 0);
        std::vector<uint64_t> b;
        if(D == m) b = gen_dense_mat(n, m, eng);
        else b = gen_mat(n, m, D, eng);
        //std::cout << "Generator matrix: " << std::endl;
        //for(int i = 0; i < n; i++) std::cout << binstring(b[i], m) << std::endl;
        std::uniform_int_distribution<uint64_t>  distr(0, (1ull<<m)-1ull);
        uint64_t target = distr(eng);
        //std::cout << "target = " << binstring(target,m) << std::endl;
        uint64_t gray = 0;
        uint64_t oldgray = 0;
        uint64_t diff;
        uint64_t codetarg = target;
        size_t index;
        uint64_t t,tmax;
        tmax = 1ull<<n;
        for(t = 0; t < tmax; t++) {
            oldgray = gray;
            gray = t ^ (t >> 1);
            diff = oldgray ^ gray;
            index = std::countr_zero(diff);
            count[std::popcount(codetarg)]++;
            codetarg ^= b[index];
        }
        allcounts.push_back(count);
    }
    std::cout << fws("h") << fws("avg(W(h))") << fws("stddev(W(h))") << std::endl;
    std::vector<double>avgs(m+1);
    std::vector<double>stddevs(m+1);
    for(size_t h = 0; h <= m; h++) {
        double avg = 0.0;
        double msd = 0.0;
        double delta;
        for(int trial = 0; trial < trials; trial++) avg += (double)allcounts[trial][h];
        avg /= (double)trials;
        for(int trial = 0; trial < trials; trial++) {
            delta = (double)allcounts[trial][h] - avg;
            msd += delta*delta;
        }
        msd /= (double)trials;
        avgs[h] = avg;
        stddevs[h] = sqrt(msd);
        std::cout << fwf(h) << fwf(avg) << fwf(sqrt(msd)) << std::endl;
    }
    std::string outname = "raw" + std::to_string(n) + "_" + std::to_string(m) + "_" + std::to_string(D) + ".tsv";
    std::ofstream outfile(outname);
    outfile << "seed = " << seed << std::endl;
    outfile << fws("h");
    for(int h = 0; h <= m; h++) outfile << fwi(h);
    outfile << std::endl;
    for(int trial = 0; trial < trials; trial++) {
        outfile << fws("trial" + std::to_string(trial));
        for(int h = 0; h <= m; h++) outfile << fwf(allcounts[trial][h]);
        outfile << std::endl;
    }
    outfile << fws("avg");
    for(int h = 0; h <= m; h++) outfile << fwf(avgs[h]);
    outfile << std::endl << fws("stddev");
    for(int h = 0; h <= m; h++) outfile << fwf(stddevs[h]);
    outfile.close();
    return 0;
}


/*int main() {
    uint16_t t;
    uint16_t gray = 0;
    uint16_t oldgray = 0;
    uint16_t diff;
    int hits[256];
    for(t = 0; t <= 255; t++) hits[t] = 0;
    for(t = 0; t <= 255; t++) {
        oldgray = gray;
        gray = t ^ (t >> 1);
        diff = oldgray ^ gray;
        std::cout << (int)t << '\t' << (std::bitset<8>)t << '\t' << (std::bitset<8>)gray << '\t' << (std::bitset<8>)diff << '\t' << std::countr_zero(diff) << std::endl;
        hits[gray]++;
        if(t != 0 && std::popcount(diff) != 1) std::cout << "Error!" << std::endl;
    }
    for(t = 0; t <= 255; t++) {
        if(hits[t] != 1) std::cout << "Error!" << std::endl;
    }
    return 0;
}*/