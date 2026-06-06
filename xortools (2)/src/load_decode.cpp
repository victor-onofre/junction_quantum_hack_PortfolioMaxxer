#include <vector>
#include <random>
#include <sstream>
#include <fstream>
#include "xorsat.hpp"
#include "utils.hpp"
#include "bitmatrix.hpp"
#include "ldpc.hpp"

void save_results(char *input_name, std::vector<double> success_rate) {
    std::ostringstream buffer;
    buffer << input_name << ".out";
    std::string outname = buffer.str();
    std::ofstream outfile(outname);
    for(const auto &r : success_rate) outfile << r << "\t";
    outfile.close();
}

int main(int argc, char *argv[]) {
    if(argc < 2 || argc > 5) {
        std::cout << "Usage: load_decode instance.tsv <seed> <errors> <trials>" << std::endl;
        return 0;
    }
    instance I;
    if(!load_instance(argv[1], I)) return 0;
    std::random_device rd;
    uint64_t seed = rd();
    if(argc >= 3) seed = std::stoull(argv[2]);
    std::mt19937_64 eng(seed);
    std::cout << "seed = " << seed << std::endl;
    bitmatrix H, G;
    H = parity_check_matrix(I);
    bp dec(H);
    if(H.num_cols() < 1E5) generator_matrix(H, G); //too costly to compute if H is huge
    std::vector<double> success_rates;
    int success;
    double success_rate = 1.0;
    double p_flip;
    int emax = 0;
    size_t errors = 0;
    bitvector random_codeword;
    int reps = 10;
    if(argc == 5) {
        int raw = std::stoi(argv[4]);
        if(raw < 1 || raw > 1E6) {
            std::cout << "Invalid reps" << std::endl;
            return 0;
        }
        reps = raw;
    }
    if(argc >= 4) {
        int raw = std::stoi(argv[3]);
        if(raw < 0 || raw > (int)H.num_cols()) {
            std::cout << "Invalid error count." << std::endl;
            return 0;
        }
        errors = (size_t)raw;
        success = 0;
        p_flip = (double)errors/(double)H.num_cols();
        for(int repetition = 0; repetition < reps; repetition++) {
            if(H.num_cols() < 1E5) random_codeword = rand_word(G, eng);
            else random_codeword.zeros(H.num_cols()); //not so random, but definitely a codeword
            bitvector corrupted_codeword = apply_rand_err(random_codeword, eng, errors);
            bitvector decoded = dec.bp_decode(corrupted_codeword, p_flip);
            if(decoded == random_codeword) success++;
        }
        success_rate = (double)success/(double)reps;
        std::cout << argv[1] << "\t" << seed << "\t" << errors << "\t" << success_rate << std::endl;
    }
    else {
        while(errors < H.num_cols() && success_rate >= 0.01) {
            success = 0;
            std::cout << errors << ": ";
            p_flip = (double)errors/(double)H.num_cols();
            int max_success_iter = 0;
            for(int repetition = 0; repetition < reps; repetition++) {
                if(H.num_cols() < 1E5) random_codeword = rand_word(G, eng);
                else random_codeword.zeros(H.num_cols()); //not so random, but definitely a codeword
                bitvector corrupted_codeword = apply_rand_err(random_codeword, eng, errors);
                bitvector decoded = dec.bp_decode(corrupted_codeword, p_flip);
                if(decoded == random_codeword) {
                    success++;
                    if(dec.iterations() > max_success_iter) max_success_iter = dec.iterations();
                }
            }
            success_rate = (double)success/(double)reps;
            std::cout << success_rate << "\t(" << max_success_iter << ")" << std::endl;
            if(success_rate >= 0.6) emax = errors;
            success_rates.push_back(success_rate);
            errors++;
        }
        std::cout << std::endl << "emax = " << emax << std::endl;
        save_results(argv[1], success_rates);
    }
    return 0;
}
