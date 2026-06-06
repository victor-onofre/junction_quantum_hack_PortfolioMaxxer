#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include "bitmatrix.hpp"
#include "utils.hpp"
#include "xorsat.hpp"
#include "ldpc.hpp"
#include "gurobi_c++.h"

bool bitround(double x) {
    if(x >= 0.5) return true;
    return false;
}

GRBVar fetchvar(GRBModel *model, std::string name, int i = -1, int j = -1, int k = -1) {
    std::string label = name;
    if(i >= 0) label += "_" + std::to_string(i);
    if(j >= 0) label += "_" + std::to_string(j);
    if(k >= 0) label += "_" + std::to_string(k);
    return model->getVarByName(label);
}

class lp {
    public:
        lp(const bitmatrix &parity_check_matrix);
        ~lp();
        bitvector lp_decode(const bitvector &corrupted_codeword, double p_flip);
        void printstats();
        int iterations(); //returns the number of iterations used in the most recent decode
        //TODO: implement this later
        //bitvector lp_decode(const bitvector &corrupted_codeword, std::vector <double> p_flip);
    private:
        bitmatrix H;
        GRBEnv env;
        GRBModel *model;
        size_t bits;
        size_t checks;
};

void lp::printstats() {
    int num_ip_vars = model->get(GRB_IntAttr_NumVars);
    int num_ip_cons = model->get(GRB_IntAttr_NumConstrs);
    std::cout << "Integer Linear Program created with: " << std::endl;
    std::cout << num_ip_vars << " ILP variables" << std::endl;
    std::cout << num_ip_cons << " ILP constraints" << std::endl;
}

/*
   This is the model called IPD in:
   Akin Tanatmis, Stefan Ruzika, Horst W. Hamacher, Mayur Punekar, Frank Kienle, and Norbert Wehn
   A Separation Algorithm for Improved LP-Decoding of Linear Block Codes
   IEEE Transactions on Information Theory 56, no. 7 (2010): 3277-3289.
*/
lp::lp(const bitmatrix &parity_check_matrix) {
    H = parity_check_matrix;
    model = new GRBModel(&env);
    model->set(GRB_IntParam_LogToConsole, 0);
    GRBLinExpr sum;
    bits = H.num_cols();
    checks = H.num_rows();
    //as a placeholder, initialize the objective function as if we are given the all-zeros bit string
    //and a bit flip probability of 0.1;
    double gamma = log(0.1/0.9);
    //Nbit[i] is the set of indices for the parity checks neighboring bit i
    std::vector<std::vector<size_t>>Nbit(bits);
    //Ncheck[j] is the set of indices for the bits neighboring parity check j
    std::vector<std::vector<size_t>>Ncheck(checks);
    //T[j] is the set of even numbers from 0 to 2*floor[neighb(j)/2], where neighb(j) is the number of bits neighboring check j
    std::vector<std::vector<size_t>>T(checks);
    for(size_t i = 0; i < bits; i++) {
        bitvector colvec = parity_check_matrix.col_vec(i);
        Nbit[i] = colvec.to_sparse();
    }
    for(size_t j = 0; j < checks; j++) {
        bitvector rowvec = parity_check_matrix.row_vec(j);
        Ncheck[j] = rowvec.to_sparse();
    }
    //add the variables
    std::string label;
    for(size_t i = 0; i < bits; i++) {
        label = "x_" + std::to_string(i);
        model->addVar(0.0,1.0,gamma,GRB_BINARY,label); //includes dummy objective all +gamma
    }
    for(size_t j = 0; j < checks; j++) {
        label = "z_" + std::to_string(j);
        model->addVar(0,bits,0.0,GRB_INTEGER,label); //slack variable needn't be bigger than dimension of parity check matrix
    }
    model->update(); //so that getVarByName will work in fetchvar
    //add the constraints
    for(size_t j = 0; j < checks; j++) {
        sum = 0;
        for(size_t i : Ncheck[j]) sum += fetchvar(model,"x",i);
        sum -= 2 * fetchvar(model,"z", j);
        model->addConstr(sum == 0);
    }
    model->update();
}

lp::~lp() {
    delete model;
}

bitvector lp::lp_decode(const bitvector &corrupted_codeword, double p_flip) {
    bitvector d(corrupted_codeword.size());
    GRBLinExpr obj;
    if(p_flip < 1.0E-6) {
        p_flip = 1.0E-6;
        notify("Warning: regularizing p_flip to 1.0E-6.");
    }
    if(p_flip > 1.0 - 1.0E-6) {
        p_flip = 1.0 - 1.0E-6;
        notify("Warning: regularizing p_flip to 1.0-1.0E-6.");
    }
    double gamma = log(p_flip/(1.0-p_flip));
    for(size_t i = 0; i < bits; i++) {
        if(corrupted_codeword.get(i)) obj += gamma * fetchvar(model, "x", i);
        else obj -= gamma * fetchvar(model, "x", i);
    }
    model->setObjective(obj, GRB_MINIMIZE);
    model->optimize();
    for(size_t i = 0; i < d.size(); i++) d.set(i,bitround(fetchvar(model, "x", i).get(GRB_DoubleAttr_X)));
    return d;
}

int lp::iterations() {
    return (int)model->get(GRB_DoubleAttr_IterCount);
}

void save_results(char *input_name, std::vector<double> success_rate) {
    std::ostringstream buffer;
    buffer << input_name << ".ipout";
    std::string outname = buffer.str();
    std::ofstream outfile(outname);
    for(const auto &r : success_rate) outfile << r << "\t";
    outfile.close();
}

int main(int argc, char *argv[]) {
    if(argc < 2 || argc > 5) {
        std::cout << "Usage: load_decode_ip instance.tsv <seed> <errors> <trials>" << std::endl;
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
    lp dec(H);
    dec.printstats();
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
            bitvector decoded = dec.lp_decode(corrupted_codeword, p_flip);
            if(decoded == random_codeword) success++;
        }
        success_rate = (double)success/(double)reps;
        std::cout << argv[1] << "\t" << seed << "\t" << errors << "\t" << success_rate << std::endl;
    }
    else {
        while(errors < H.num_cols() && success_rate >= 0.5) {
            success = 0;
            std::cout << errors << ": ";
            p_flip = (double)errors/(double)H.num_cols();
            int max_success_iter = 0;
            for(int repetition = 0; repetition < reps; repetition++) {
                if(H.num_cols() < 1E5) random_codeword = rand_word(G, eng);
                else random_codeword.zeros(H.num_cols()); //not so random, but definitely a codeword
                bitvector corrupted_codeword = apply_rand_err(random_codeword, eng, errors);
                bitvector decoded = dec.lp_decode(corrupted_codeword, p_flip);
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
