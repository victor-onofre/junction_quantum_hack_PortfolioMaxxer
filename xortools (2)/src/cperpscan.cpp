#include <iostream>
#include <bitset>
#include <bit>
#include <cstdint>
#include <random>
#include <vector>
#include <map>
#include <fstream>
#include "bitmatrix.hpp"
#include "ldpc.hpp"
#include "utils.hpp"
#include "xorsat.hpp"

std::string binstring(uint64_t x, size_t n) {
    std::bitset<64> allbits;
    allbits = (std::bitset<64>)x;
    std::string returnval(n, '0');
    for(size_t i = 0; i < n; i++) if(allbits[i]) returnval[i] = '1';
    return returnval;
}

bool valid(bitmatrix H, bitmatrix G, int k, int D) {
    for(size_t i = 0; i < H.num_rows(); i++) {
        if(H.row_vec(i).count() != D) return false;
    }
    for(size_t j = 0; j < H.num_cols(); j++) {
        if(H.col_vec(j).count() != k) return false;
    }
    bitmatrix prod = G * H.transpose();
    for(size_t i = 0; i < prod.num_rows(); i++) {
        if(prod.row_vec(i).count() != 0) return false;
    }
    return true;
}

//parity check matrix from the Gallager ensemble
bitmatrix check_mat_gallager(size_t k, size_t D, size_t bsize, std::mt19937_64 &eng) {
    int m = D*bsize;
    int n = k*bsize;
    bitmatrix B(n,m);
    std::vector<int> perm(m);
    for(size_t i = 0; i < m; i++) perm[i] = i;
    for(size_t a = 0; a < k; a++) {
        std::shuffle(perm.begin(), perm.end(), eng);
        for(size_t b = 0; b < bsize; b++) {
            size_t row_index = a*bsize+b;
            for(size_t entry_index = 0; entry_index < D; entry_index++) {
                B.set(row_index, perm[bsize*entry_index + row_index%bsize], 1);
            }
        }
    }
    return B;
}

//generator matrix from the Gallager ensemble
std::vector<uint64_t> gen_mat_gallager(size_t k, size_t D, size_t bsize, std::mt19937_64 &eng) {
    int m = D*bsize;
    if(D < k || m > 64) {
        std::cout << "Error: bad parameters in gen_mat_gallager" << std::endl;
        return {};
    }
    bitmatrix H = check_mat_gallager(k, D, bsize, eng);
    bitmatrix G;
    generator_matrix(H, G);
    std::cout << "H\n" << H << std::endl;
    std::cout << "G\n" << G << std::endl;
    if(!valid(H, G, k, D)) std::cout << "Bad!" << std::endl;
    else std::cout << "Good: " << G.num_rows() << std::endl;
    std::vector <uint64_t> returnval(G.num_rows()); //dimension is m-n if H is full rank
    for(size_t i = 0; i < G.num_rows(); i++) returnval[i] = G.row_vec(i).to_num();
    return returnval;
}

//If min is O(n) then this method takes O(n^2). One might be able to do better
//by replacing std::vector<size_t> zeros with a heap or something.
void upweight_rows(bitmatrix &B, int min, std::mt19937_64 &eng) {
    if(min > B.num_cols()) {
        std::cout << "Error: invalid parameters in upweight" << std::endl;
        return;
    }
    for(size_t row_index = 0; row_index < B.num_rows(); row_index++) {
        int initial_weight = B.row_vec(row_index).count();
        if(initial_weight >= min) continue;
        std::vector<size_t> zeros;
        for(size_t i = 0; i < B.num_cols(); i++) if(!B.get(row_index,i)) zeros.push_back(i);
        for(int pass = 0; pass < min - initial_weight; pass++) {
            size_t n = zeros.size();
            std::uniform_int_distribution<int> randindex(0,n-1);
            size_t zero_index = randindex(eng);
            int flip_index = zeros[zero_index];
            B.set(row_index, flip_index, 1);
            zeros.erase(zeros.begin()+zero_index);
        }
    }
}

void upweight_rows_and_cols(bitmatrix &B, int min, std::mt19937_64 &eng) {
    upweight_rows(B, min, eng);
    bitmatrix C = B.transpose();
    upweight_rows(C, min, eng);
    B = C.transpose();
}

//parity check matrix from the random_sparse ensemble
bitmatrix check_mat_sparse(size_t n, size_t m, double p, int min, std::mt19937_64 &eng) {
    bitmatrix B(n,m);
    std::bernoulli_distribution bd(p);
    std::uniform_int_distribution<size_t> rowzero(0,n-2);
    std::uniform_int_distribution<size_t> colzero(0,m-2);
    for(size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < m; j++) {
            if(bd(eng)) B.set(i,j,1);
        }
    }
    upweight_rows_and_cols(B, min, eng);
    return B;
}

bool good_sparse(const bitmatrix &H, double &density, int min) {
    bool returnval = true;
    size_t total_count = 0;
    for(size_t row_index = 0; row_index < H.num_rows(); row_index++) {
        size_t count = H.row_vec(row_index).count();
        total_count += count;
        if(count < min) { std::cout << "Row " << row_index << " has weight " << count << std::endl; returnval = false; }
    }
    for(size_t col_index = 0; col_index < H.num_cols(); col_index++) {
        size_t count = H.col_vec(col_index).count();
        if(count < min) { std::cout << "Col " << col_index << " has weight " << count << std::endl; returnval = false; }
    }
    double numerator = (double)total_count;
    double denominator = (double)(H.num_rows() * H.num_cols());
    density = numerator/denominator;
    return returnval;
}

//generator matrix from my random-sparse ensemble
std::vector<uint64_t> gen_mat_sparse(size_t n, size_t m, double p, std::mt19937_64 &eng) {
    if(m > 64) {
        std::cout << "Error: bad parameters in gen_mat_sparse" << std::endl;
        return {};
    }
    bitmatrix H = check_mat_sparse(n,m,p,3,eng);
    bitmatrix G;
    generator_matrix(H, G);
    //std::cout << "H\n" << H << std::endl;
    //std::cout << "G\n" << G << std::endl;
    double density;
    bool good, sound;
    good = good_sparse(H,density,3);
    bitmatrix prod = G * H.transpose();
    sound = true;
    for(size_t i = 0; i < prod.num_rows(); i++) {
        if(prod.row_vec(i).count() != 0) {
            sound = false;
            break;
        }
    }
    if(!good) std::cout << "Bad!" << std::endl;
    if(!sound) std::cout << "Unsound!" << std::endl;
    if(good && sound) std::cout << "Good: " << G.num_rows() << ": " << density << std::endl;
    std::vector <uint64_t> returnval(G.num_rows()); //dimension is m-n if H is full rank
    for(size_t i = 0; i < G.num_rows(); i++) returnval[i] = G.row_vec(i).to_num();
    return returnval;
}

std::vector<uint64_t> scanweights(const std::vector<uint64_t> &b, size_t m) {
    std::vector<uint64_t> count(m+1,0);
    size_t n = b.size();
    if(n > 40) {
        notify("Cannot scan more than 2^40 codewords.");
        return {};
    }
    uint64_t gray = 0;
    uint64_t oldgray = 0;
    uint64_t codeword = 0;
    uint64_t diff;
    size_t index;
    uint64_t t,tmax;
    tmax = 1ull<<n;
    count[0] = 1;
    for(t = 1; t < tmax; t++) {
        oldgray = gray;
        gray = t ^ (t >> 1);
        diff = oldgray ^ gray;
        index = std::countr_zero(diff);
        codeword ^= b[index];
        count[std::popcount(codeword)]++;
    }
    return count;
}

void printstats(const std::vector<std::vector<uint64_t>> &allcounts) {
    int trials = allcounts.size();
    int m = allcounts[0].size() - 1;
    std::cout << fws("h") << fws("avg(W(h))") << fws("stddev(W(h))") << std::endl;
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
        std::cout << fwf(h) << fwf(avg) << fwf(sqrt(msd)) << std::endl;
    }
}

int main(int argc, char *argv[]) {
    if(argc != 5 && argc != 3) {
        std::cout << "Usage: cperpscan G k D bsize (Gallager random-regular)\n";
        std::cout << "Or:    cperpscan S m n p (Sparse random)\n";
        std::cout << "Or:    cperpscan L filename (load code)" << std::endl;
        return 0;
    }
    std::random_device rd;
    uint64_t seed = rd();
    std::mt19937_64 eng(seed);
    std::vector<std::vector<uint64_t>> allcounts;
    int trials = 10;
    std::string type = argv[1];
    if(type == "S") {
        std::cout << "seed = " << seed << std::endl;
        int m = std::atoi(argv[2]);
        int checks = std::atoi(argv[3]);
        double p = std::atof(argv[4]);
        std::cout << "trials " << trials << "\nchecks " << checks << "\nm " << m << "\np " << p << std::endl;
        for(int trial = 0; trial < trials; trial++) {
            std::vector<uint64_t> b = gen_mat_sparse(checks, m, p, eng);
            std::vector<uint64_t> count = scanweights(b, m);
            allcounts.push_back(count);
        }
        printstats(allcounts);
        return 0;
    }
    if(type == "G") {
        std::cout << "seed = " << seed << std::endl;
        int k = std::atoi(argv[2]);
        int D = std::atoi(argv[3]);
        int bsize = std::atoi(argv[4]);
        std::cout << "trials " << trials << "\nk " << k << "\nD " << D << "\nbsize " << bsize << std::endl;
        for(int trial = 0; trial < trials; trial++) {
            std::vector<uint64_t> b = gen_mat_gallager(k, D, bsize, eng);
            std::vector<uint64_t> count = scanweights(b, D*bsize);
            allcounts.push_back(count);
        }
        printstats(allcounts);
        return 0;
    }
    if(type == "L") {
        instance I;
        if(!load_instance(argv[2], I)) return 0;
        bitmatrix H = parity_check_matrix(I);
        bitmatrix G;
        generator_matrix(H, G);
        std::cout << "H:\n" << H << std::endl;
        std::cout << "G:\n" << G << std::endl;
        std::vector<uint64_t>b(G.num_rows());
        for(size_t i = 0; i < G.num_rows(); i++) b[i] = G.row_vec(i).to_num();
        std::vector<uint64_t> count = scanweights(b, G.num_cols());
        for(int w = 0; w <= G.num_cols(); w++) std::cout << fwi(w) << fwi(count[w]) << std::endl;
        return 0;
    }
    std::cout << "Unrecognized type: " << type << std::endl;
    return 0;
}

/*int main() {
    uint16_t t;
    uint16_t gray = 0;
    uint16_t oldgray = 0;
    uint16_t diff;
    int hits[256];
    for(t = 0; t <= 255; t++) hits[t] = 0;
    for(t = 1; t <= 255; t++) {
        oldgray = gray;
        gray = t ^ (t >> 1);
        diff = oldgray ^ gray;
        std::cout << (int)t << '\t' << (std::bitset<8>)oldgray << '\t' << (std::bitset<8>)gray << '\t' << (std::bitset<8>)diff << '\t' << std::countr_zero(diff) << std::endl;
        hits[oldgray]++;
        if(t != 0 && std::popcount(diff) != 1) std::cout << "Error!" << std::endl;
    }
    hits[gray]++;
    for(t = 0; t <= 255; t++) {
        if(hits[t] != 1) std::cout << "Error!" << std::endl;
    }
    return 0;
}*/